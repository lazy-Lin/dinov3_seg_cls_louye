@echo off
REM 大数据集（1w+ 样本）训练脚本 - Windows 版本
REM 使用 DINOv3-B/14，三阶段训练

echo ==========================================
echo 大数据集瑕疵检测模型训练
echo 数据量: 1w+ 样本
echo 模型: DINOv3-B/14 (~94M 参数)
echo ==========================================

REM 配置
set DATA_ROOT=data/your_10k_dataset
set BACKBONE=dinov3_vitb14
set IMAGE_SIZE=518

REM 检查数据目录
if not exist "%DATA_ROOT%" (
    echo 错误: 数据目录不存在: %DATA_ROOT%
    echo 请先准备数据或修改 DATA_ROOT 变量
    pause
    exit /b 1
)

REM ============================================
REM 阶段 1: 冻结骨干网络 (30 epochs)
REM ============================================
echo.
echo ==========================================
echo 阶段 1: 冻结骨干网络训练
echo 目标: 快速训练分类和分割头
echo 预期: 分类准确率 90-93%%, 分割 IoU 0.70-0.75
echo ==========================================

python defect_detection_project/scripts/train_defect_model.py --data_root %DATA_ROOT% --backbone %BACKBONE% --image_size %IMAGE_SIZE% --freeze_backbone --batch_size 32 --epochs 30 --lr 1e-3 --weight_decay 0.01 --dropout 0.2 --use_dynamic_weights --save_dir checkpoints/stage1_frozen --num_workers 4

if errorlevel 1 (
    echo 阶段 1 训练失败！
    pause
    exit /b 1
)

echo.
echo ✓ 阶段 1 完成！
echo 模型保存在: checkpoints/stage1_frozen/

REM ============================================
REM 阶段 2: 微调骨干网络 (70 epochs)
REM ============================================
echo.
echo ==========================================
echo 阶段 2: 微调整个网络
echo 目标: 端到端优化，提升性能
echo 预期: 分类准确率 96-99%%, 分割 IoU 0.80-0.90
echo ==========================================

python defect_detection_project/scripts/train_defect_model.py --data_root %DATA_ROOT% --backbone %BACKBONE% --image_size %IMAGE_SIZE% --batch_size 16 --epochs 70 --lr 5e-5 --weight_decay 0.05 --dropout 0.2 --use_dynamic_weights --save_dir checkpoints/stage2_finetune --num_workers 4

if errorlevel 1 (
    echo 阶段 2 训练失败！
    pause
    exit /b 1
)

echo.
echo ✓ 阶段 2 完成！
echo 模型保存在: checkpoints/stage2_finetune/

REM ============================================
REM 阶段 3: 不确定性加权优化 (20 epochs)
REM ============================================
echo.
echo ==========================================
echo 阶段 3: 不确定性加权微调（可选）
echo 目标: 自动平衡多任务损失
echo 预期: 分类准确率 97-99%%, 分割 IoU 0.82-0.92
echo ==========================================

set /p STAGE3="是否执行阶段 3？(y/n): "
if /i "%STAGE3%"=="y" (
    python defect_detection_project/scripts/train_defect_model.py --data_root %DATA_ROOT% --backbone %BACKBONE% --image_size %IMAGE_SIZE% --batch_size 16 --epochs 20 --lr 1e-5 --weight_decay 0.05 --dropout 0.2 --use_uncertainty_weighting --save_dir checkpoints/stage3_uncertainty --num_workers 4
    
    if errorlevel 1 (
        echo 阶段 3 训练失败！
        pause
        exit /b 1
    )
    
    echo.
    echo ✓ 阶段 3 完成！
    echo 模型保存在: checkpoints/stage3_uncertainty/
    set BEST_MODEL=checkpoints/stage3_uncertainty/best_accuracy.pth
) else (
    echo 跳过阶段 3
    set BEST_MODEL=checkpoints/stage2_finetune/best_accuracy.pth
)

REM ============================================
REM 推理测试
REM ============================================
echo.
echo ==========================================
echo 推理测试
echo ==========================================

if exist "data/test_images" (
    echo 在测试集上运行推理...
    python examples/inference_defect_model.py --checkpoint %BEST_MODEL% --backbone %BACKBONE% --image_size %IMAGE_SIZE% --image_dir data/test_images --output_dir results/final
    
    echo.
    echo ✓ 推理完成！
    echo 结果保存在: results/final/
) else (
    echo 未找到测试集目录 data/test_images，跳过推理
)

REM ============================================
REM 完成
REM ============================================
echo.
echo ==========================================
echo 训练完成！
echo ==========================================
echo.
echo 模型文件：
echo   - 阶段 1: checkpoints/stage1_frozen/best_accuracy.pth
echo   - 阶段 2: checkpoints/stage2_finetune/best_accuracy.pth
if /i "%STAGE3%"=="y" (
    echo   - 阶段 3: checkpoints/stage3_uncertainty/best_accuracy.pth
)
echo.
echo 推荐使用: %BEST_MODEL%
echo.
echo 下一步：
echo   1. 查看训练日志和指标
echo   2. 在测试集上评估性能
echo   3. 可视化预测结果
echo.

pause
