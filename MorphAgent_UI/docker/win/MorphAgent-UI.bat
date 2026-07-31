@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Packaged zip: compose next to this .bat
REM Repo layout: docker\win\ → parent is docker\
set "COMPOSE_DIR=%cd%"
if exist "%COMPOSE_DIR%\docker-compose.yml" if exist "%COMPOSE_DIR%\Dockerfile" goto have_compose
if exist "%COMPOSE_DIR%\..\docker-compose.yml" (
  pushd "%COMPOSE_DIR%\.."
  set "COMPOSE_DIR=%cd%"
  popd
  goto have_compose
)
echo ERROR: Cannot find docker-compose.yml next to this launcher.
pause
exit /b 1

:have_compose
set "UI_URL=http://localhost:6080/vnc.html?autoconnect=true&resize=remote"
if not defined DOCKER_PLATFORM set "DOCKER_PLATFORM=linux/amd64"

where docker >nul 2>&1
if errorlevel 1 (
  echo Docker is not installed.
  echo Install Docker Desktop for Windows, start it, then double-click again:
  echo https://www.docker.com/products/docker-desktop/
  start "" "https://www.docker.com/products/docker-desktop/"
  pause
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is not running.
  echo Please start Docker Desktop, wait until it is ready, then double-click again.
  pause
  exit /b 1
)

echo ============================================================
echo  MorphAgent UI (Docker^)
echo  platform: %DOCKER_PLATFORM%
echo  compose:  %COMPOSE_DIR%\docker-compose.yml
echo ============================================================
echo First launch builds the image (several minutes^). Later starts are fast.
echo.

cd /d "%COMPOSE_DIR%"
docker compose -f "%COMPOSE_DIR%\docker-compose.yml" up -d --build
if errorlevel 1 (
  echo.
  echo docker compose failed. See messages above.
  pause
  exit /b 1
)

echo.
echo Waiting for noVNC...
set /a _tries=0
:wait_loop
set /a _tries+=1
curl -fsS "http://127.0.0.1:6080/" >nul 2>&1
if not errorlevel 1 goto ready
if %_tries% GEQ 90 goto ready
timeout /t 2 /nobreak >nul
goto wait_loop

:ready
start "" "%UI_URL%"

echo.
echo MorphAgent UI is running.
echo   Browser: %UI_URL%
echo   Stop:    docker compose -f "%COMPOSE_DIR%\docker-compose.yml" down
echo.
echo Tip: Home -^> Load a previous run -^> completed_demo_run (no API key needed^).
echo.
pause
endlocal
