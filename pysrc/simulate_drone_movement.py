import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# Matplotlibでの日本語表示のための設定 (環境に応じてフォントを指定してください)
# 例:
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'MS Gothic', 'TakaoPGothic', 'IPAexGothic', 'Noto Sans CJK JP']

plt.ion() # インタラクティブモードをオンにする

# --- 1. PyTorchによる簡易障害物認識モデルの定義 ---

class SimpleObstacleDetector(nn.Module):
    def __init__(self, input_dim=8):
        super(SimpleObstacleDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def prepare_sensor_data_for_model(sensor_area_values, expected_dim=8):
    
    processed_data = []
    for val in sensor_area_values:
        if val == 1:
            processed_data.append(1.0)
        elif val == -1:
            processed_data.append(0.0)
        else:
            processed_data.append(0.0)
    while len(processed_data) < expected_dim:
        processed_data.append(0.0)
    return processed_data[:expected_dim]

INPUT_DIM_FOR_MODEL = 8
obstacle_detector_model = SimpleObstacleDetector(input_dim=INPUT_DIM_FOR_MODEL)
print(f"簡易障害物検出モデル（入力次元: {INPUT_DIM_FOR_MODEL}）が定義されました。")
print("このモデルは、概念実証（PoC）として、基本的な機械学習のワークフローを示します。")


# --- 2. シミュレーション環境の設定 ---
# (変更なし - OBSTACLE_COUNT は前回の50のまま)
GRID_SIZE = 20
OBSTACLE_COUNT = 50
MODEL_SAVE_PATH = "simple_obstacle_detector_trained.pth"

grid = np.zeros((GRID_SIZE, GRID_SIZE))
drone_pos_initial = [0, 0]
target_pos = [GRID_SIZE - 1, GRID_SIZE - 1]

def setup_environment():
    global grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    print(f"{OBSTACLE_COUNT}個の障害物を配置します。")
    for _ in range(OBSTACLE_COUNT):
        while True:
            ox, oy = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if grid[ox, oy] == 0 and (ox, oy) != tuple(drone_pos_initial) and (ox, oy) != tuple(target_pos):
                grid[ox, oy] = 1
                break
    grid[drone_pos_initial[0], drone_pos_initial[1]] = 2
    grid[target_pos[0], target_pos[1]] = 3
    print("シミュレーション環境が設定されました。")

# --- 3. 概念実証のためのモデル学習フェーズ ---

def train_conceptual_model(model, current_grid_env, epochs=20, learning_rate=0.005, samples_per_epoch=100):
    print("\n--- 概念実証のためのモデル学習開始 ---")
    print("目的: ドローン周囲のセンサー情報から、障害物の有無を予測するモデルを学習します。")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        for _ in range(samples_per_epoch):
            temp_drone_r, temp_drone_c = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            temp_sensor_input_values = []
            actual_obstacle_present_in_vicinity = False
            for dr_t in [-1, 0, 1]:
                for dc_t in [-1, 0, 1]:
                    if dr_t == 0 and dc_t == 0: continue
                    nr_t, nc_t = temp_drone_r + dr_t, temp_drone_c + dc_t
                    if 0 <= nr_t < GRID_SIZE and 0 <= nc_t < GRID_SIZE:
                        cell_value = current_grid_env[nr_t, nc_t]
                        temp_sensor_input_values.append(cell_value)
                        if cell_value == 1:
                            actual_obstacle_present_in_vicinity = True
                    else:
                        temp_sensor_input_values.append(-1)
            model_input_list = prepare_sensor_data_for_model(temp_sensor_input_values, INPUT_DIM_FOR_MODEL)
            X_train = torch.FloatTensor(model_input_list).unsqueeze(0)
            y_train = torch.FloatTensor([[1.0 if actual_obstacle_present_in_vicinity else 0.0]])
            optimizer.zero_grad()
            prediction_prob = model(X_train)
            loss = criterion(prediction_prob, y_train)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            predicted_label = (prediction_prob.item() > 0.5)
            actual_label = (y_train.item() > 0.5)
            if predicted_label == actual_label:
                correct_predictions += 1
            total_samples +=1
        avg_loss = epoch_loss / samples_per_epoch
        accuracy = correct_predictions / total_samples
        if (epoch + 1) % 5 == 0 or epoch == 0:
             print(f"エポック [{epoch+1}/{epochs}], 平均損失: {avg_loss:.4f}, 訓練精度: {accuracy:.2%}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"モデルの学習が完了し、重みが '{MODEL_SAVE_PATH}' に保存されました。")
    print("--- モデル学習終了 ---")


# --- 4. ドローンの動作ロジックとシミュレーション実行 ---
fig, ax = None, None
im_artist = None 
cbar = None 

# プロット更新時の待機時間
NORMAL_PAUSE_DURATION = 0.05
EVASION_PAUSE_DURATION = 0.2 # 回避行動時の待機時間を少し長くする

def plot_grid_interactive(current_grid_to_plot, title="ドローンシミュレーション", pause_duration=NORMAL_PAUSE_DURATION):
    """グリッドの状態をインタラクティブに描画する"""
    global fig, ax, im_artist, cbar

    if fig is None: 
        fig, ax = plt.subplots(figsize=(8, 8))
        im_artist = ax.imshow(current_grid_to_plot, cmap='viridis', origin='lower', vmin=0, vmax=4)
        ax.set_xticks(np.arange(GRID_SIZE))
        ax.set_yticks(np.arange(GRID_SIZE))
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        cmap_obj = plt.cm.viridis
        norm_obj = plt.Normalize(vmin=0, vmax=4)
        ticks_values = [0.4*i + 0.2 for i in range(5)]
        mappable = plt.cm.ScalarMappable(norm=norm_obj, cmap=cmap_obj)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, ticks=ticks_values, orientation='vertical')
        cbar.ax.set_yticklabels(['空きマス(0)', '障害物(1)', 'ドローン(2)', '目的地(3)', '移動パス(4)'], fontsize=10)

    im_artist.set_data(current_grid_to_plot)
    ax.set_title(title, fontsize=16)
    
    plt.draw()
    fig.canvas.flush_events()
    plt.pause(pause_duration) # 指定された時間だけ一時停止


def simulate_drone_movement(model_to_use, initial_grid):
    """
    学習済みモデルを使用したドローンの移動と障害物検出、経路計画のシミュレーション。
    """
    current_grid_sim = np.copy(initial_grid)
    current_pos = list(drone_pos_initial)
    path_taken = [tuple(current_pos)]

    print("\n--- ドローンシミュレーション開始（学習済みAIモデル使用） ---")
    plot_grid_interactive(current_grid_sim, "初期状態 (AIモデル使用)")

    steps = 0
    max_steps = GRID_SIZE * GRID_SIZE * 2

    model_to_use.eval()

    while current_pos != target_pos and steps < max_steps:
        steps += 1
        current_pause_duration = NORMAL_PAUSE_DURATION # 通常のプロット更新間隔
        print(f"\n--- ステップ {steps} --- 現在位置: {current_pos}")

        sensor_input_values = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = current_pos[0] + dr, current_pos[1] + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    sensor_input_values.append(current_grid_sim[nr, nc])
                else:
                    sensor_input_values.append(-1)
        
        model_input_list = prepare_sensor_data_for_model(sensor_input_values, INPUT_DIM_FOR_MODEL)
        model_input_tensor = torch.FloatTensor(model_input_list).unsqueeze(0)

        with torch.no_grad():
            prediction_prob = model_to_use(model_input_tensor)
        
        obstacle_probability = prediction_prob.item()
        obstacle_detected_by_ai = (obstacle_probability > 0.5) 

        if obstacle_detected_by_ai:
            print(f"🚨 AIモデルが障害物を検出しました (予測確率: {obstacle_probability:.2f})！経路を再計画します。明確な回避を試みます。")
            current_pause_duration = EVASION_PAUSE_DURATION # 回避時は少し長く待機
            
            evasive_move_found = False
            # 優先的な回避行動：まず上下左右を試す
            # (dr, dc, move_description)
            cardinal_moves = [
                (0, 1, "上へ回避"), (0, -1, "下へ回避"), 
                (1, 0, "右へ回避"), (-1, 0, "左へ回避")
            ]
            random.shuffle(cardinal_moves) # 回避方向の優先度をランダムにする

            for dr_evade, dc_evade, move_desc in cardinal_moves:
                next_r, next_c = current_pos[0] + dr_evade, current_pos[1] + dc_evade
                if 0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE and \
                   current_grid_sim[next_r, next_c] not in [1, 2, 4]: # 障害物、現在地、パス以外
                    
                    print(f"明確な回避行動: {move_desc}")
                    current_grid_sim[current_pos[0], current_pos[1]] = 4
                    current_pos = [next_r, next_c]
                    current_grid_sim[current_pos[0], current_pos[1]] = 2
                    path_taken.append(tuple(current_pos))
                    print(f"ドローンが {current_pos} へ移動しました ({move_desc})。")
                    evasive_move_found = True
                    break # 回避行動成功

            if not evasive_move_found:
                # 上下左右に有効な回避経路がなければ、従来通り斜めも含む8方向で探す
                print("明確な上下左右の回避経路なし。広範囲で再探索します。")
                best_next_pos_candidate = None
                min_dist_to_target = float('inf')
                # 斜めも含む8方向
                all_possible_moves = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
                random.shuffle(all_possible_moves)

                for dr_move, dc_move in all_possible_moves:
                    next_r, next_c = current_pos[0] + dr_move, current_pos[1] + dc_move
                    if 0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE and \
                       current_grid_sim[next_r, next_c] not in [1, 2, 4]:
                        dist = np.sqrt((next_r - target_pos[0])**2 + (next_c - target_pos[1])**2)
                        if dist < min_dist_to_target:
                            min_dist_to_target = dist
                            best_next_pos_candidate = [next_r, next_c]
                
                if best_next_pos_candidate:
                    current_grid_sim[current_pos[0], current_pos[1]] = 4
                    current_pos = best_next_pos_candidate
                    current_grid_sim[current_pos[0], current_pos[1]] = 2
                    path_taken.append(tuple(current_pos))
                    print(f"ドローンが {current_pos} へ移動しました (広範囲探索での回避)。")
                else:
                    print("最適な回避経路が見つかりませんでした。ドローンは停止します。")
                    break
        else: 
            # (障害物なしの場合のロジックは変更なし)
            print(f"✅ AIモデルは障害物を検出しませんでした (予測確率: {obstacle_probability:.2f})。目的地へ直進します。")
            next_r_ideal, next_c_ideal = current_pos[0], current_pos[1]
            if target_pos[0] > current_pos[0]: next_r_ideal += 1
            elif target_pos[0] < current_pos[0]: next_r_ideal -= 1
            if target_pos[1] > current_pos[1]: next_c_ideal += 1
            elif target_pos[1] < current_pos[1]: next_c_ideal -= 1
            
            if 0 <= next_r_ideal < GRID_SIZE and 0 <= next_c_ideal < GRID_SIZE and \
               current_grid_sim[next_r_ideal, next_c_ideal] != 1: 
                current_grid_sim[current_pos[0], current_pos[1]] = 4 
                current_pos = [next_r_ideal, next_c_ideal]
                current_grid_sim[current_pos[0], current_pos[1]] = 2 
                path_taken.append(tuple(current_pos))
                print(f"ドローンが {current_pos} へ移動しました。")
            else:
                print(f"直進予定先({next_r_ideal},{next_c_ideal})に問題あり。簡易回避を試みます。")
                moved_randomly = False
                possible_moves_straight_fallback = [(0,1), (0,-1), (1,0), (-1,0)] 
                random.shuffle(possible_moves_straight_fallback)
                for dr_fallback, dc_fallback in possible_moves_straight_fallback:
                    temp_r, temp_c = current_pos[0] + dr_fallback, current_pos[1] + dc_fallback
                    if 0 <= temp_r < GRID_SIZE and 0 <= temp_c < GRID_SIZE and \
                       current_grid_sim[temp_r, temp_c] not in [1, 2, 4]: 
                        current_grid_sim[current_pos[0], current_pos[1]] = 4
                        current_pos = [temp_r, temp_c]
                        current_grid_sim[current_pos[0], current_pos[1]] = 2
                        path_taken.append(tuple(current_pos))
                        print(f"ドローンが {current_pos} へランダム回避移動しました。")
                        moved_randomly = True
                        break
                if not moved_randomly:
                    print("ランダム回避も失敗。ドローンは停止します。")
                    break 
        
        plot_grid_interactive(current_grid_sim, f"ステップ {steps} - ドローン位置: {current_pos}", pause_duration=current_pause_duration)

    if tuple(current_pos) == tuple(target_pos):
        print(f"\n✅ ドローンが目的地に到達しました！ 総ステップ数: {steps}")
    else:
        print(f"\n❌ ドローンが目的地に到達できませんでした。総ステップ数: {steps}")
    
    print("--- シミュレーション終了 ---")
    plt.ioff() 
    plot_grid_interactive(current_grid_sim, f"最終状態 - ステップ {steps}", pause_duration=NORMAL_PAUSE_DURATION) 
    print("最終状態のプロットを表示しています。ウィンドウを閉じるとプログラムが終了します。")
    plt.show() 

# --- メイン処理 ---
if __name__ == "__main__":
    RANDOM_SEED = 42
    print(f"使用する乱数シード: {RANDOM_SEED}")
    print(f"障害物の数 (OBSTACLE_COUNT): {OBSTACLE_COUNT}")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    setup_environment()
    train_conceptual_model(obstacle_detector_model, grid, epochs=30, samples_per_epoch=200)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n学習済みモデル '{MODEL_SAVE_PATH}' を読み込んでいます...")
        try:
            obstacle_detector_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            print("モデルの読み込みに成功しました。")
        except Exception as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            print("学習済みの重みなしでシミュレーションを続行します（性能は期待できません）。")
    else:
        print(f"学習済みモデルファイル '{MODEL_SAVE_PATH}' が見つかりません。")
        print("学習済みの重みなしでシミュレーションを続行します（性能は期待できません）。")

    simulate_drone_movement(obstacle_detector_model, grid)

    print("\nこのスクリプトは、Python、NumPy、PyTorch、Matplotlibを用いた機械学習プロジェクトの")
    print("基本的な要素（データ準備、モデル定義、学習、推論、評価、可視化）を示しています。")
    print("AI開発企業において、顧客の課題解決に貢献できる")
    print("機械学習エンジニアとしての素養を示す一助となれば幸いです。")
    print("特に、DXコンサルタントとしてのプロジェクト推進経験（要件定義から評価・改善まで）は、")
    print("技術とビジネスを繋ぐ役割において強みになると考えております。")

