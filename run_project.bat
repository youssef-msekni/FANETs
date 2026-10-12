@echo off
REM ======================================================================
REM run_project.bat -- lance tout le pipeline P5 en une seule commande :
REM   1. active le venv
REM   2. lance la simulation (main.py) -> genere comparison_results.json
REM   3. lance plot_comparison.py -> genere les PNG dans charts/
REM   4. demarre un serveur local et ouvre Dashboard.html dans le navigateur
REM
REM A placer dans le dossier FANETs/myagent/ (a cote de main.py) et lancer
REM en double-cliquant, ou via `run_project.bat` dans un terminal.
REM ======================================================================

cd /d "%~dp0"

echo ===== Preparation du venv =====
if not exist venv\Scripts\python.exe (
    echo Creation du venv...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo ===== Etape 1/3 : Simulation (main.py) =====
python main.py
if errorlevel 1 (
    echo.
    echo [ERREUR] La simulation a echoue -- voir le message ci-dessus.
    pause
    exit /b 1
)

echo.
echo ===== Etape 2/3 : Graphiques (plot_comparison.py) =====
python plot_comparison.py
if errorlevel 1 (
    echo.
    echo [ERREUR] La generation des graphiques a echoue -- voir le message ci-dessus.
    pause
    exit /b 1
)

echo.
echo ===== Etape 3/3 : Dashboard interactif =====
echo Demarrage du serveur local sur http://localhost:8000 ...
start "FANET dashboard server" cmd /k python -m http.server 8000

REM laisse le temps au serveur de demarrer avant d'ouvrir le navigateur
timeout /t 2 /nobreak >nul
start http://localhost:8000/Dashboard.html

echo.
echo ===== Termine =====
echo Le dashboard s'actualise automatiquement toutes les 3 secondes.
echo Pour rejouer la simulation : relancez simplement "python main.py" puis
echo "python plot_comparison.py" dans un AUTRE terminal (laissez le serveur
echo ouvert) -- le dashboard deja ouvert se mettra a jour tout seul.
echo.
echo Fermez la fenetre "FANET dashboard server" pour arreter le serveur local.
pause
