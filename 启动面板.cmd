@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 尚未创建本项目的 Python 虚拟环境。
    echo 请在此目录打开终端，依次执行：
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -e ".[ui,validation]" -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo 安装完成后再次双击此文件。
    pause
    exit /b 1
)
echo 本地面板地址：http://127.0.0.1:8501
echo 当前窗口运行本地服务。关闭此窗口即可停止。
".venv\Scripts\python.exe" -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=false --browser.gatherUsageStats=false
if errorlevel 1 (
    echo 启动失败，请按 README 安装界面依赖。
    pause
)
