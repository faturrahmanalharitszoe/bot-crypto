@echo off
:: =====================================================================
:: Deployment Script for Binance Crypto Bot
:: Transfers all updated files and models to the VPS.
:: Modify the VPS_IP and REMOTE_DIR variables if they are different.
:: =====================================================================

set VPS_IP=38.47.180.84
set REMOTE_DIR=/home/backend/bot-crypto

echo 📡 Preparing to upload files to VPS (%VPS_IP%)...
echo --------------------------------------------------

:: 1. Copy root & bot files
echo 📤 Uploading configuration and code files...
scp bot\trader.py root@%VPS_IP%:%REMOTE_DIR%/bot/
scp bot\exchange.py root@%VPS_IP%:%REMOTE_DIR%/bot/
scp bot\pair_selector.py root@%VPS_IP%:%REMOTE_DIR%/bot/
scp bot\web_dashboard.py root@%VPS_IP%:%REMOTE_DIR%/bot/
scp bot\ml_model.py root@%VPS_IP%:%REMOTE_DIR%/bot/
scp bot\nn_model.py root@%VPS_IP%:%REMOTE_DIR%/bot/
scp config.py root@%VPS_IP%:%REMOTE_DIR%/
scp train_model.py root@%VPS_IP%:%REMOTE_DIR%/
scp requirements.txt root@%VPS_IP%:%REMOTE_DIR%/

:: 2. Copy model weights & scalers
echo 📤 Uploading model weights and scaler...
scp models\dl_model.pth root@%VPS_IP%:%REMOTE_DIR%/models/
scp models\scaler.pkl root@%VPS_IP%:%REMOTE_DIR%/models/

echo --------------------------------------------------
echo ✅ Upload completed successfully!
echo 🚀 Next step: SSH to VPS, run "pip install -r requirements.txt", and "pm2 restart bot_crypto".
pause
