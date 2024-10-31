
"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Soyong Shin, Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import math
import os
import pickle
from skel.alignment.losses import compute_anchor_pose, compute_anchor_trans, compute_pose_loss, compute_scapula_loss, compute_spine_loss, compute_time_loss, pretty_loss_print
from skel.alignment.utils import location_to_spheres, to_numpy, to_params, to_torch
import torch
from tqdm import trange
import smplx
import torch.nn.functional as F
from psbody.mesh import Mesh, MeshViewer, MeshViewers
import skel.config as cg
from skel.skel_model import SKEL
import omegaconf 

# SKELとSMPLモデルを使って、人体の3Dポーズをフィッティングするためのクラス
class SkelFitter(object):
    
    def __init__(self, gender, device, num_betas=10, export_meshes=False, config_path=None) -> None:
        # 初期化関数: SMPLおよびSKELモデルを生成し、フィッティングマスクなどの初期設定を行う

        # SMPLモデルの作成
        self.smpl = smplx.create(cg.smpl_folder, model_type='smpl', gender=gender, num_betas=num_betas, batch_size=1, export_meshes=False).to(device)
        # SKELモデルの作成
        self.skel = SKEL(gender).to(device)
        self.gender = gender
        self.device = device
        self.num_betas = num_betas
        
        # フィッティングマスクをロードし、頂点ごとのマスクを設定
        fitting_indices = pickle.load(open(cg.fitting_mask_file, 'rb'))
        fitting_mask = torch.zeros(6890, dtype=torch.bool, device=self.device)
        fitting_mask[fitting_indices] = 1
        self.fitting_mask = fitting_mask.reshape(1, -1, 1).to(self.device) # 頂点に対して適用するマスクの形にする (1xVx1)
        
        # トルソ（胴体）の頂点マスクを設定
        smpl_torso_joints = [0,3]
        verts_mask = (self.smpl.lbs_weights[:,smpl_torso_joints]>0.5).sum(dim=-1)>0
        self.torso_verts_mask = verts_mask.unsqueeze(0).unsqueeze(-1) # 頂点がBxVx3の形状であるため、調整
        
        self.export_meshes = export_meshes

        # 設定ファイルのパスの設定
        if config_path is None:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(package_directory, 'config.yaml')
            
        self.cfg =  omegaconf.OmegaConf.load(config_path)
           
        # メッシュビューワーのインスタンス化
        if('DISABLE_VIEWER' in os.environ):
            self.mv = None
            print("\n DISABLE_VIEWER flag is set, running in headless mode")
        else:
            self.mv = MeshViewers((1,2),  keepalive=self.cfg.keepalive_meshviewer)
        
        
    def run_fit(self, 
            trans_in, 
            betas_in, 
            poses_in, 
            batch_size=20, 
            skel_data_init=None, 
            force_recompute=False, 
            debug=False,
            watch_frame=0,
            freevert_mesh=None):
        """SKELをSMPLシーケンスにフィットさせる"""

        # シーケンスのフレーム数とウォッチフレームの設定
        self.nb_frames = poses_in.shape[0]
        self.watch_frame = watch_frame
        if self.watch_frame >= self.nb_frames:
            raise ValueError(f'watch_frame {self.watch_frame} is larger than the number of frames {self.nb_frames}. Please provide a watch frame index smaller than the number of frames ({self.nb_frames})')
            
        self.is_skel_data_init = skel_data_init is not None
        self.force_recompute = force_recompute
        
        print('Fitting {} frames'.format(self.nb_frames))
        print('Watching frame: {}'.format(watch_frame))
         
        # SKELの初期パラメータを初期化
        body_params = self._init_params(betas_in, poses_in, trans_in, skel_data_init)
    
        # 全シーケンスをバッチに分割して並列最適化を行う
        if batch_size > self.nb_frames:
            batch_size = self.nb_frames
            print('Batch size is larger than the number of frames. Setting batch size to {}'.format(batch_size))
            
        n_batch = math.ceil(self.nb_frames/batch_size)
        pbar = trange(n_batch, desc='Running batch optimization')
        
        # フレームごとのSKELパラメータの結果を保存するための辞書を初期化
        out_keys = ['poses', 'betas', 'trans'] 
        if self.export_meshes:
            out_keys += ['skel_v', 'skin_v', 'smpl_v']
        res_dict = {key: [] for key in out_keys}
        
        res_dict['gender'] = self.gender
        if self.export_meshes:
            res_dict['skel_f'] = self.skel.skel_f.cpu().numpy().copy()
            res_dict['skin_f'] = self.skel.skin_f.cpu().numpy().copy()
            res_dict['smpl_f'] = self.smpl.faces
     
        # 全バッチを順番にフィットさせる
        for i in pbar:  
                
            if debug:
                # デバッグモードでは最初のバッチのみを実行
                if i > 1:
                    continue
            
            # バッチの開始・終了インデックスを取得
            i_start =  i * batch_size
            i_end = min((i+1) * batch_size, self.nb_frames)

            # バッチのフィッティング
            betas, poses, trans, verts = self._fit_batch(body_params, i, i_start, i_end)
            
            # 結果を保存
            res_dict['poses'].append(poses)
            res_dict['betas'].append(betas)
            res_dict['trans'].append(trans)
            if self.export_meshes:
                # メッシュの頂点を保存
                skel_output = self.skel.forward(poses=poses, betas=betas, trans=trans, poses_type='skel', skelmesh=True)
                res_dict['skel_v'].append(skel_output.skel_verts)
                res_dict['skin_v'].append(skel_output.skin_verts)
                res_dict['smpl_v'].append(verts)
                
            # 次のフレームを現在のフレームで初期化
            body_params['poses_skel'][i_end:] = poses[-1:]
            body_params['trans_skel'][i_end:] = trans[-1]
            body_params['betas_skel'][i_end:] = betas[-1:]
            
        # バッチを結合してnumpy配列に変換
        for key, val in res_dict.items():
            if isinstance(val, list):
                res_dict[key] = torch.cat(val, dim=0).detach().cpu().numpy()
                
        return res_dict
        
    def _init_params(self, betas_smpl, poses_smpl, trans_smpl, skel_data_init=None):
        """ SMPLデータ辞書とオプションのSKELデータ辞書から初期SKELパラメータを返す """
    
        # SMPLパラメータを準備
        betas_smpl = to_torch(betas_smpl, self.device)
        poses_smpl = to_torch(poses_smpl, self.device)
        trans_smpl = to_torch(trans_smpl, self.device)
        
        if skel_data_init is None or self.force_recompute:
            # SKELのパラメータがない場合や再計算が必要な場合
            poses_skel = torch.zeros((self.nb_frames, self.skel.num_q_params), device=self.device)
            poses_skel[:, :3] = poses_smpl[:, :3] # グローバルな向きはSMPLとSKELで共通しているので初期化
            poses_skel[:, 0] = -poses_smpl[:, 0] # SKELでの軸定義が異なるため、符号を変更

            betas_skel = torch.zeros((self.nb_frames, 10), device=self.device)
            betas_skel[:] = betas_smpl[..., :10]
            
            trans_skel = trans_smpl # トランスレーションもSMPLとSKELで共通なのでそのまま使用
            
        else:
            # 以前のアラインメントから読み込み
            betas_skel = to_torch(skel_data_init['betas'], self.device)
            poses_skel = to_torch(skel_data_init['poses'], self.device)
            trans_skel = to_torch(skel_data_init['trans'], self.device)
            
        # 必要なボディパラメータを辞書にまとめる
        body_params = {
            'betas_skel': betas_skel,
            'poses_skel': poses_skel,
            'trans_skel': trans_skel,
            'betas_smpl': betas_smpl,
            'poses_smpl': poses_smpl,
            'trans_smpl': trans_smpl
        }

        return body_params
        
    def _fit_batch(self, body_params_in, i, i_start, i_end):
        """ 各バッチ用のパラメータを作成し、最適化を実行 """
        
        # バッチのサンプリング
        assert body_params_in['betas_smpl'].shape[0] == 1, f"beta_smpl should be of shape 1xF where F is the number of frames, got {body_params_in['betas_smpl'].shape}"
        body_params = { key: val[i_start:i_end] for key, val in body_params_in.items() if key != 'betas_smpl'} # betas_smplは全フレーム共通のため無視
        body_params['betas_smpl'] = body_params_in['betas_smpl'].clone()

        # SMPLパラメータの設定
        betas_smpl = body_params['betas_smpl']
        poses_smpl = body_params['poses_smpl']
        trans_smpl = body_params['trans_smpl']
        
        # SKELパラメータの設定
        betas = to_params(body_params['betas_skel'], device=self.device)
        poses = to_params(body_params['poses_skel'], device=self.device)
        trans = to_params(body_params['trans_skel'], device=self.device)
        
        if 'verts' in body_params:
            verts = body_params['verts']
        else:
            # SMPLの前向き計算を実行し、SMPLのボディ頂点を取得
            smpl_output = self.smpl(betas=betas_smpl, body_pose=poses_smpl[:, 3:], transl=trans_smpl, global_orient=poses_smpl[:, :3])
            verts = smpl_output.vertices
            
        # 最適化の実行
        config = self.cfg.optim_steps
        current_cfg = config[0]
        
        if not self.is_skel_data_init:
            # 初回フィッティングではグローバル回転とトランスレーションの最適化のみを行う
            print(f'Step 0: {current_cfg.description}')
            self._optim([trans, poses], poses, betas, trans, verts, current_cfg)

        # その後の最適化ステップを実行
        for ci, cfg in enumerate(config[1:]):
            current_cfg.update(cfg)
            print(f'Step {ci+1}: {current_cfg.description}')
            self._optim([poses], poses, betas, trans, verts, current_cfg)
        
        return betas, poses, trans, verts
    
    def _optim(self, params, poses, betas, trans, verts, cfg):
        # SMPLの頂点から解剖学的な関節を回帰
        anat_joints = torch.einsum('bik,ji->bjk', [verts, self.skel.J_regressor_osim]) 
        dJ = torch.zeros((poses.shape[0], 24, 3), device=betas.device)
        
        # オプティマイザの作成
        optimizer = torch.optim.LBFGS(params, 
                                      lr=cfg.lr, 
                                      max_iter=cfg.max_iter, 
                                      line_search_fn=cfg.line_search_fn,  
                                      tolerance_change=cfg.tolerance_change)
        
        poses_init = poses.detach().clone()               
        trans_init = trans.detach().clone()

        # 損失関数を使用して最適化を行うクロージャ関数の定義
        def closure():
            optimizer.zero_grad()
            
            fi = self.watch_frame # 表示するバッチ内のフレーム
            output = self.skel.forward(poses=poses[fi:fi+1], 
                                        betas=betas[fi:fi+1], 
                                        trans=trans[fi:fi+1], 
                                        poses_type='skel', 
                                        dJ=dJ[fi:fi+1],
                                        skelmesh=True)
        
            # 各ステップでのプロット
            self._fstep_plot(output, cfg, verts[fi:fi+1], anat_joints[fi:fi+1])
                
            # 損失を計算
            loss_dict = self._fitting_loss(poses, poses_init, betas, trans, trans_init, dJ, anat_joints, verts, cfg)
            print(pretty_loss_print(loss_dict))
            
            # 総損失を計算して逆伝播
            loss = sum(loss_dict.values())                     
            loss.backward()
        
            return loss

        # 指定された回数だけ最適化ステップを実行
        for step_i in range(cfg.num_steps):
            loss = optimizer.step(closure).item()
    
    def _fitting_loss(self, poses, poses_init, betas, trans, trans_init, dJ, anat_joints, verts, cfg):
        loss_dict = {}
        
        # マスクを取得し、ポーズの一部をマスクする
        pose_mask, verts_mask, joint_mask = self._get_masks(cfg) 
        poses = poses * pose_mask + poses_init * (1 - pose_mask)

        # 出力を計算して損失を求める
        output = self.skel.forward(poses=poses, betas=betas, trans=trans, poses_type='skel', dJ=dJ, skelmesh=False)
        
        # SMPLの頂点をフィッティング
        loss_dict['verts_loss_loose'] = cfg.l_verts_loose * (verts_mask * (output.skin_verts - verts) ** 2).sum() / (((verts_mask).sum() * self.nb_frames))

        # 関節の一致度
        loss_dict['joint_loss'] = cfg.l_joint * (joint_mask * (output.joints - anat_joints) ** 2).mean()

        # 時間的な一貫性を保持する損失
        if poses.shape[0] > 1:
            loss_dict['time_loss'] = cfg.l_time_loss * F.mse_loss(poses[1:], poses[:-1])
        
        # 基本的なポーズ損失
        loss_dict['pose_loss'] = cfg.l_pose_loss * compute_pose_loss(poses, poses_init)

        # 基本的な損失を使わない場合はアンカーポーズやスケープラの損失を追加
        if cfg.use_basic_loss is False:
            loss_dict['anch_rot'] = cfg.l_anch_pose * compute_anchor_pose(poses, poses_init)
            loss_dict['anch_trans'] = cfg.l_anch_trans * compute_anchor_trans(trans, trans_init)
            loss_dict['verts_loss'] = cfg.l_verts * (verts_mask * self.fitting_mask * (output.skin_verts - verts) ** 2).sum() / (self.fitting_mask * verts_mask).sum()
            loss_dict['scapula_loss'] = cfg.l_scapula_loss * compute_scapula_loss(poses)
            loss_dict['spine_loss'] = cfg.l_spine_loss * compute_spine_loss(poses)
            
            for key in ['scapula_loss', 'spine_loss', 'pose_loss']:
                loss_dict[key] = cfg.pose_reg_factor * loss_dict[key]
        
        return loss_dict

    def _fstep_plot(self, output, cfg, verts, anat_joints):
        """ 各ステップでのプロットを行う関数 """
        
        if('DISABLE_VIEWER' in os.environ):
            return
        
        pose_mask, verts_mask, joint_mask = self._get_masks(cfg) 
        
        skin_err_value = ((output.skin_verts[0] - verts[0]) ** 2).sum(dim=-1).sqrt()
        skin_err_value = skin_err_value / 0.05
        skin_err_value = to_numpy(skin_err_value)
        
        skin_mesh = Mesh(v=to_numpy(output.skin_verts[0]), f=[], vc='white')
        skel_mesh = Mesh(v=to_numpy(output.skel_verts[0]), f=self.skel.skel_f.cpu().numpy(), vc='white')
        
        # SMPLの頂点にエラーカラーを表示
        smpl_verts = to_numpy(verts[0])
        smpl_mesh = Mesh(v=smpl_verts, f=self.smpl.faces)
        smpl_mesh.set_vertex_colors_from_weights(skin_err_value, scale_to_range_1=False)       
        
        smpl_mesh_masked = Mesh(v=smpl_verts[to_numpy(verts_mask[0,:,0])], f=[], vc='green')
        smpl_mesh_pc = Mesh(v=smpl_verts, f=[], vc='green')
        
        skin_mesh_err = Mesh(v=to_numpy(output.skin_verts[0]), f=self.skel.skin_f.cpu().numpy(), vc='white')
        skin_mesh_err.set_vertex_colors_from_weights(skin_err_value, scale_to_range_1=False) 
        # 表示するメッシュリスト
        meshes_left = [skin_mesh_err, smpl_mesh_pc, skel_mesh]
        meshes_right = [smpl_mesh_masked, skin_mesh, skel_mesh]

        if cfg.l_joint > 0:
            # 関節をプロット
            meshes_right += location_to_spheres(to_numpy(output.joints[joint_mask[:,:,0]]), color=(1,0,0), radius=0.02)
            meshes_right += location_to_spheres(to_numpy(anat_joints[joint_mask[:,:,0]]), color=(0,1,0), radius=0.02)

        self.mv[0][0].set_dynamic_meshes(meshes_left)
        self.mv[0][1].set_dynamic_meshes(meshes_right)
