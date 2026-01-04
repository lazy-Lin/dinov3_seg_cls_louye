@echo off
REM 瑕疵检测模型快速开始脚本 (Windows)

echo ==========================================
echo 瑕疵检测模型 - 快速开始
echo ==========================================

REM 1. 创建示例数据
echo.
echo [1/4] 创建示例数据...
python examples/prepare_defect_data.py dummy --output_dir data/defect_demo --num_defect 100 --num_normal 100

REM 2. 训练模型
echo.
echo [2/4] 开始训练...
python examples/train_defect_model.py --data_root data/defect_demo --backbone dinov3_vits14 --batch_size 16 --epochs 10 --lr 1e-3 --freeze_backbone --use_dynamic_weights --save_dir checkpoints/demo

REM 3. 推理测试
echo.
echo [3/4] 运行推理...
python examples/inference_defect_model.py --checkpoint checkpoints/demo/best_accuracy.pth --backbone dinov3_vits14 --image_dir data/defect_demo/images --output_dir results/demo

REM 4. 完成
echo.
echo [4/4] 完成！
echo 训练模型保存在: checkpoints/demo/
echo 推理结果保存在: results/demo/
echo.
echo 查看结果：
echo   - 可视化: results/demo/*.png
echo   - 摘要: results/demo/results_summary.txt

pause
