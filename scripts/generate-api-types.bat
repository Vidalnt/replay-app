@echo off
cd /d "%~dp0\.."
echo Downloading OpenAPI spec from http://127.0.0.1:62362/openapi.json...
curl -s "http://127.0.0.1:62362/openapi.json" -o shared/openapi.json
echo Generating TypeScript types...
npx --yes openapi-client-axios-typegen shared/openapi.json -t > shared/clients/types.d.ts
if %errorlevel% neq 0 (
    echo Failed to generate TypeScript types.
    pause
    exit /b 1
)
echo API types generated successfully!
pause
