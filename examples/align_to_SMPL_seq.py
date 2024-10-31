# 必要なモジュールをインポート
import argparse  # コマンドライン引数を扱うためのモジュール
import os  # ファイルやディレクトリの操作用モジュール
import pickle  # オブジェクトのシリアライズ・デシリアライズ用モジュール
import trimesh  # 3Dメッシュ操作用のモジュール

# カスタムモジュールから関数・クラスをインポート
from skel.alignment.aligner import SkelFitter  # 骨格フィッタークラス
from skel.alignment.utils import load_smpl_seq  # SMPLシーケンスをロードする関数

# メインの処理がここから始まる
if __name__ == '__main__':
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='Align SKEL to a SMPL sequence')
    # 引数を定義
    parser.add_argument('smpl_seq_path', type=str, help='Path to the SMPL sequence')
    parser.add_argument('-o', '--out_dir', type=str, help='Output directory', default='output')
    parser.add_argument('-F', '--force-recompute', help='Force recomputation of the alignment', action='store_true')
    parser.add_argument('-D', '--debug', help='Only run the fit on the first minibatch to test', action='store_true')
    parser.add_argument('-B', '--batch_size', type=int, help='Batch size', default=3000)
    parser.add_argument('-w', '--watch_frame', type=int, help='Frame of the batch to display', default=0)
    parser.add_argument('--gender', type=str, help='Gender of the subject (only needed if not provided with smpl_data_path)', default='female')
    parser.add_argument('-m', '--export_meshes', choices=[None, 'mesh', 'pickle'], default=None, 
                        help='If not None, export the resulting meshes (skin and skeleton), either as .obj files or as a pickle file containing the vertices and faces')
    parser.add_argument('--config', help='Yaml config file containing parameters for training. You can create a config tailored to align a specific sequence. When left to None, the default config will be used', default=None)
    
    # 引数をパース
    args = parser.parse_args()
    
    # SMPLシーケンスをロード
    smpl_seq = load_smpl_seq(args.smpl_seq_path, gender=args.gender, straighten_hands=False)
    
    # 出力ディレクトリを作成
    subj_name = os.path.basename(args.smpl_seq_path).split(".")[0]  # 入力ファイル名からサブジェクト名を取得
    subj_dir = os.path.join(args.out_dir, subj_name)  # 出力ディレクトリのパス
    os.makedirs(subj_dir, exist_ok=True)  # 出力ディレクトリがなければ作成
    pkl_path = os.path.join(subj_dir, subj_name+'_skel.pkl')  # 保存するpickleファイルのパス
    
    # 以前のアライメント結果があれば初期化用にロード
    if os.path.exists(pkl_path) and not args.force_recompute:
        print('Previous aligned SKEL sequence found at {}. Will be used as initialization.'.format(pkl_path))
        skel_data_init = pickle.load(open(pkl_path, 'rb'))  # 既存のデータを初期化用にロード
    else:
        skel_data_init = None  # 初期データがなければNoneに設定
    
    # SkelFitterクラスのインスタンスを作成
    skel_fitter = SkelFitter(smpl_seq['gender'], 
                             device='cuda:0',  # GPUを使用（CUDAデバイス指定）
                             export_meshes=args.export_meshes is not None,
                             config_path=args.config)
    
    # アライメントの実行
    skel_seq = skel_fitter.run_fit(smpl_seq['trans'], 
                               smpl_seq['betas'], 
                               smpl_seq['poses'], 
                               batch_size=args.batch_size,
                               skel_data_init=skel_data_init,  # 初期化用のSKELデータ
                               force_recompute=args.force_recompute,
                               debug=args.debug,  # デバッグオプション
                               watch_frame=args.watch_frame)  # 特定のフレームを表示するためのオプション
    
    # メッシュをエクスポートするオプションが有効な場合
    if args.export_meshes == 'mesh':
        # pklファイルからメッシュデータを削除して軽量化
        skel_seq = {key: val for key, val in skel_seq.items() if key not in ['skel_v', 'skel_f', 'skin_v', 'skin_f', 'smpl_v', 'smpl_f']}
        
        # 必要なフォルダを作成
        for folder in ['SKEL_skin', 'SKEL_skel', 'SMPL']:
            os.makedirs(os.path.join(subj_dir, 'meshes', folder), exist_ok=True)
        
        # メッシュをエクスポートする
        mesh_folder = os.path.join(subj_dir, 'meshes')
        for i in range(skel_seq['skel_v'].shape[0]):
            # Skinメッシュをエクスポート
            skin_mesh = trimesh.Trimesh(vertices=skel_seq['skin_v'][i], faces=skel_seq['skin_f'])
            skin_mesh.export(os.path.join(mesh_folder, 'SKEL_skin', f'skel_skin_{i}.obj'))
            
            # Skelメッシュをエクスポート
            skel_mesh = trimesh.Trimesh(vertices=skel_seq['skel_v'][i], faces=skel_seq['skel_f'])
            skel_mesh.export(os.path.join(mesh_folder, 'SKEL_skel', f'skel_skel{i}.obj'))
            
            # SMPLメッシュをエクスポート
            smpl_mesh = trimesh.Trimesh(vertices=skel_seq['smpl_v'][i], faces=skel_seq['smpl_f'])
            smpl_mesh.export(os.path.join(mesh_folder, 'SMPL', f'smpl_{i}.obj'))
    
    # SKELシーケンスをpickleファイルとして保存
    pickle.dump(skel_seq, open(pkl_path, 'wb'))
    print('Saved aligned SKEL sequence to {}'.format(pkl_path))
