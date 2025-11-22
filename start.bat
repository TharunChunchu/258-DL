@echo off
echo Starting Face Recognition System...

cd /d "%~dp0"

REM Kill existing processes on ports 5001 and 3000
echo Killing existing processes on ports 5001 and 3000...

REM Kill Flask backend on port 5001
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5001 ^| findstr LISTENING') do (
    echo Killing process on port 5001...
    taskkill /F /PID %%a >nul 2>&1
)

REM Kill React frontend on port 3000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do (
    echo Killing process on port 3000...
    taskkill /F /PID %%a >nul 2>&1
)

timeout /t 2 /nobreak >nul

REM Check if frontend node_modules exists
if not exist "frontend\node_modules" (
    echo Installing React dependencies...
    cd frontend
    call npm install
    cd ..
)

REM Start Flask backend
echo Starting Flask backend on port 5001...
start "Flask Backend" cmd /k "python pytorch_app.py"

timeout /t 3 /nobreak >nul

REM Start React frontend
echo Starting React frontend on port 3000...
cd frontend
start "React Frontend" cmd /k "npm start"
cd ..

echo.
echo ========================================
echo Both servers are starting!
echo ========================================
echo Flask Backend:  http://localhost:5001
echo React Frontend: http://localhost:3000
echo ========================================
echo.
echo Close the command windows to stop the servers
pause

