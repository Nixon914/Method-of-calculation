import numpy as np

def spring_design_cost_verification():
    
    # 標準最佳解 (來自 Scholarpedia)
    x1_optimal = 	0.0516597413  # 線徑 (d)
    x2_optimal = 0.3560125  # 平均線圈直徑 (D)  
    x3_optimal = 11.3304429494  # 有效圈數 (N)
    
    print(f"標準最佳解:")
    print(f"  d (線徑) = {x1_optimal:.6f} 英寸")
    print(f"  D (平均線圈直徑) = {x2_optimal:.6f} 英寸")
    print(f"  N (有效圈數) = {x3_optimal:.6f}")
    
    # 計算目標函數 - 彈簧重量
    cost_optimal = (x3_optimal + 2) * x2_optimal * x1_optimal**2
    
    print(f"\n目標函數值 (彈簧重量):")
    print(f"  f(x*) = (N + 2) × D × d² = {cost_optimal:.6f}")
    
    # 計算約束條件
    print(f"\n約束條件檢查:")
    
    # 約束1: 剪切應力約束
    g1 = 1 - x3_optimal * x2_optimal**3 / (71785 * x1_optimal**4)
    print(f"  g1 (剪切應力) = {g1:.8f} {'✓' if g1 <= 0 else '✗'}")
    
    # 約束2: 挫屈約束  
    g2 = (4 * x2_optimal**2 - x1_optimal * x2_optimal) / (12566 * (x2_optimal * x1_optimal**3 - x1_optimal**4)) + 1 / (5108 * x1_optimal**2) - 1
    print(f"  g2 (挫屈約束) = {g2:.8f} {'✓' if g2 <= 0 else '✗'}")
    
    # 約束3: 撓度約束
    g3 = 1 - 140.45 * x1_optimal / (x2_optimal**2 * x3_optimal)
    print(f"  g3 (撓度約束) = {g3:.8f} {'✓' if g3 <= 0 else '✗'}")
    
    # 約束4: 空間約束
    g4 = (x1_optimal + x2_optimal) / 1.5 - 1
    print(f"  g4 (空間約束) = {g4:.8f} {'✓' if g4 <= 0 else '✗'}")
    
    # 檢查變數邊界
    print(f"\n變數邊界檢查:")
    print(f"  0.05 ≤ d ≤ 2.0:  {x1_optimal:.6f} {'✓' if 0.05 <= x1_optimal <= 2.0 else '✗'}")
    print(f"  0.25 ≤ D ≤ 1.3:  {x2_optimal:.6f} {'✓' if 0.25 <= x2_optimal <= 1.3 else '✗'}")
    print(f"  2.0 ≤ N ≤ 15.0:  {x3_optimal:.6f} {'✓' if 2.0 <= x3_optimal <= 15.0 else '✗'}")
    
    # 工程計算
    spring_index = x2_optimal / x1_optimal
    total_coils = x3_optimal + 2
    
    print(f"\n工程參數:")
    print(f"  彈簧指數 (D/d) = {spring_index:.2f} (理想範圍: 4-12)")
    print(f"  總圈數 (N + 2) = {total_coils:.2f}")
    print(f"  彈簧重量 = {cost_optimal:.6f} 磅")
    
    return cost_optimal, g1, g2, g3, g4

def compare_with_your_result():
    """
    比較標準解與您的算法結果
    """
    print("\n" + "=" * 60)
    print("結果比較")
    print("=" * 60)
    
    # 標準最佳解
    standard_cost = 0.012665
    your_cost = 0.012732  # 您的算法結果
    
    error_percentage = abs(your_cost - standard_cost) / standard_cost * 100
    
    print(f"標準最佳解 (Scholarpedia): {standard_cost:.6f}")
    print(f"您的算法結果:              {your_cost:.6f}")
    print(f"絕對誤差:                  {abs(your_cost - standard_cost):.6f}")
    print(f"相對誤差:                  {error_percentage:.2f}%")
    
    if error_percentage < 1.0:
        print("🏆 優秀！您的算法達到了世界級性能！")
    elif error_percentage < 5.0:
        print("✅ 很好！您的算法表現優異！")
    else:
        print("⚠️  需要改進算法性能")

def test_spring_function():
    """
    測試彈簧設計函數
    """
    print("\n" + "=" * 60)
    print("函數測試")
    print("=" * 60)
    
    # 使用標準最佳解測試
    x_optimal = np.array([0.051690, 0.356750, 11.28126])
    
    def spring_design_cost(x):
        x = x.flatten()
        x1, x2, x3 = x[0], x[1], x[2]
        
        # 目標函數 - 最小化彈簧重量
        cost = (x3 + 2) * x2 * x1 * x1
        
        # 約束條件計算
        g1 = 1 - x3 * x2**3 / (71785 * x1**4)
        g2 = (4 * x2**2 - x1 * x2) / (12566 * (x2 * x1**3 - x1**4)) + 1 / (5108 * x1**2) - 1
        g3 = 1 - 140.45 * x1 / (x2**2 * x3)
        g4 = (x1 + x2) / 1.5 - 1
        
        print(f"函數測試結果:")
        print(f"  目標函數值: {cost:.6f}")
        print(f"  g1 = {g1:.8f}")
        print(f"  g2 = {g2:.8f}")
        print(f"  g3 = {g3:.8f}")
        print(f"  g4 = {g4:.8f}")
        
        return cost
    
    result = spring_design_cost(x_optimal)
    return result

if __name__ == "__main__":
    # 執行驗證
    cost, g1, g2, g3, g4 = spring_design_cost_verification()
    
    # 比較結果
    compare_with_your_result()
    
    # 測試函數
    test_spring_function()
    
    print("\n" + "=" * 60)
    print("驗證完成！")
    print("=" * 60)