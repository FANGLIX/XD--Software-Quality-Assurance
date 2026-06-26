$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$py = "C:\Users\Administrator\Documents\Codex\2026-06-18\654111813-qq\.venv_real\Scripts\python.exe"
$script = "E:\软件质量保障\项目完成交付\code\har_lstm_pytorch_real_experiment.py"
$out = "E:\软件质量保障\项目完成交付\video_demo_output"

New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host "Starting quick demo run..."
Write-Host "This demo uses epochs=2, train-limit=512, test-limit=256."
Write-Host "Formal report results are stored in the delivery folder CSV file."
Write-Host ""

& $py $script --epochs 2 --train-limit 512 --test-limit 256 --batch-size 64 --output-dir $out

Write-Host ""
Write-Host "Demo run finished."
Write-Host "Output folder is video_demo_output under the project delivery folder."
Write-Host "Open video_demo_output\results_real.csv to check demo results."
Pause
