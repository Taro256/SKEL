import os
import pickle
import torch
import numpy as np
from psbody.mesh.sphere import Sphere

# to_params = lambda x: torch.from_numpy(x).float().to(self.device).requires_grad_(True)
# to_torch = lambda x: torch.from_numpy(x).float().to(self.device)

# NumPy配列をPyTorchのテンソルに変換し、指定されたデバイス上に転送し、勾配計算を有効にする関数
def to_params(x, device):
    return x.to(device).requires_grad_(True)

# NumPy配列をPyTorchのテンソルに変換し、指定されたデバイス上に転送する関数
def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)

# PyTorchテンソルをNumPy配列に変換する関数
def to_numpy(x):
    return x.detach().cpu().numpy()

# SMPLシーケンスデータをファイルからロードし、必要な形式に整形する関数
def load_smpl_seq(smpl_seq_path, gender=None, straighten_hands=False):

    # 指定されたパスが存在するか確認
    if not os.path.exists(smpl_seq_path):
        raise Exception('Path does not exist: {}'.format(smpl_seq_path))
    
    # ファイルが.pkl形式の場合、pickleを使って読み込む
    if smpl_seq_path.endswith('.pkl'):
        data_dict = pickle.load(open(smpl_seq_path, 'rb'))
    
    # ファイルが.npz形式の場合、NumPyを使って読み込む
    elif smpl_seq_path.endswith('.npz'):
        data_dict = np.load(smpl_seq_path, allow_pickle=True)
        
        # ファイルに特定のキーが含まれている場合、そのデータを取得
        if data_dict.files == ['pred_smpl_parms', 'verts', 'pred_cam_t']:
            data_dict = data_dict['pred_smpl_parms'].item()
        else:
            # 他の場合はすべてのデータを辞書に変換
            data_dict = {key: data_dict[key] for key in data_dict.keys()} 
    else:
        raise Exception('Unknown file format: {}. Supported formats are .pkl and .npz'.format(smpl_seq_path))
        
    # 必要なキーを持つ辞書を新しく作成
    data_fixed = {}  
    
    # 性別の取得または指定
    if 'gender' not in data_dict:
        assert gender is not None, f"The provided SMPL data dictionary does not contain gender, you need to pass it in command line"
        data_fixed['gender'] = gender
    elif not isinstance(data_dict['gender'], str):
        # 性別の型がstrでない場合（例えばarray形式）、文字列に変換
        data_fixed['gender'] = str(data_dict['gender'])
    else:
        data_fixed['gender'] = gender
            
    # テンソルをNumPy配列に変換
    for key, val in data_dict.items():
        if isinstance(val, torch.Tensor):
            data_dict[key] = val.detach().cpu().numpy()

    # SMPLのポーズ情報を取得
    if 'poses' in data_dict: 
        poses = data_dict['poses']
    elif 'body_pose_axis_angle' in data_dict and 'global_orient_axis_angle' in data_dict:
        poses = np.concatenate([data_dict['global_orient_axis_angle'], data_dict['body_pose_axis_angle']], axis=1)
        poses = poses.reshape(-1, 72)
    elif 'body_pose' in data_dict and 'global_orient' in data_dict and 'body_pose_axis_angle' in data_dict and 'global_orient_axis_angle' in data_dict:
        poses = np.concatenate([data_dict['global_orient_axis_angle'], data_dict['body_pose_axis_angle']], axis=-1)
    elif 'body_pose' in data_dict and 'global_orient' in data_dict:
        poses = np.concatenate([data_dict['global_orient'], data_dict['body_pose']], axis=-1)
    else: 
        raise Exception(f"Could not find poses in {smpl_seq_path}. Available keys: {data_dict.keys()})")
        
    # SMPL+Hのポーズの場合、手のポーズを除去
    if poses.shape[1] == 156:
        smpl_poses = np.zeros((poses.shape[0], 72))
        smpl_poses[:, :72-2*3] = poses[:, :72-2*3] # SMPL関節22と23のパラメータは0に設定
        poses = smpl_poses
    
    # 手のポーズをまっすぐにする場合、該当部分をゼロに設定
    if straighten_hands:      
        poses[:, 72-2*3:] = 0
        
    data_fixed['poses'] = poses
        
    # 位置の取得
    if 'trans' in data_dict:
        data_fixed['trans'] = data_dict['trans']
    elif 'transl' in data_dict:
        data_fixed['trans'] = data_dict['transl']
    else:
        print(f'WARNING: Could not find translation in {smpl_seq_path}. Setting translation to zeros.')
        data_fixed['trans'] = np.zeros((poses.shape[0], 3))

    # ベータパラメータの取得（10個に制限）
    betas = data_dict['betas'][..., :10] 
    if len(betas.shape) == 1 and len(poses.shape) == 2:
        betas = betas[None, :] # バッチ次元を追加
    data_fixed['betas'] = betas
     
    # 必須のキーが存在するか確認
    for key in ['trans', 'poses', 'betas', 'gender']:
        assert key in data_fixed.keys(), f'Could not find {key} in {smpl_seq_path}. Available keys: {data_fixed.keys()})'
        
    # 出力用の辞書を作成
    out_dict = {}
    out_dict['trans'] = data_fixed['trans']
    out_dict['poses'] = data_fixed['poses']
    out_dict['betas'] = data_fixed['betas']
    out_dict['gender'] = data_fixed['gender']
    
    # パラメータの形状を確認
    num_batches = out_dict['poses'].shape[0]
    assert out_dict['trans'].shape[0] == num_batches, f"Number of translations ({out_dict['trans'].shape[0]}) does not match the number of poses ({num_batches})"
    assert out_dict['poses'].shape[1] == 72, f"Poses should have 72 parameters, found {out_dict['poses'].shape[1]} parameters"
    assert out_dict['betas'].shape[1] == 10, f"Betas should have 10 parameters, found {out_dict['betas'].shape[1]} parameters"
    assert out_dict['gender'] in ['male', 'female'], f"Gender should be either 'male' or 'female', found {out_dict['gender']}"

    return out_dict
        
        
# 3D位置に球体を配置するための関数
def location_to_spheres(loc, color=(1,0,0), radius=0.02):
    """与えられた3Dポイントの配列に基づいて、それらの位置に球体を配置する。

    Args:
        loc (numpy.array): Nx3の3D位置を示す配列
        color (tuple, optional): 球体の色（RGBベクトル）。デフォルトは赤 (1,0,0)。
        radius (float, optional): 球体の半径（メートル単位）。デフォルトは0.02。

    Returns:
        list: 球体のメッシュのリスト
    """

    # 各位置に球体を配置して、そのリストを作成
    cL = [Sphere(np.asarray([loc[i, 0], loc[i, 1], loc[i, 2]]), radius).to_mesh() for i in range(loc.shape[0])]
    for spL in cL:
        spL.set_vertex_colors(np.array(color))  # 球体に色を設定
    return cL
