@echo off
cd C:\Users\bryan\OneDrive\Desktop\imagetotext\code_checker\

echo Creating templates folder if not exists...
if not exist templates mkdir templates

echo Creating register.html...
copy nul templates\register.html
echo Register.html created!

echo Creating login.html...
copy nul templates\login.html
echo Login.html created!

echo Creating forgot_password.html...
copy nul templates\forgot_password.html
echo Forgot_password.html created!

echo.
echo ========================================
echo Template files created successfully!
echo ========================================
echo.
echo Please copy the HTML code into each file.
echo.
pause