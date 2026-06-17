param(
  [string]$HostAddress = "127.0.0.1",
  [int]$Port = 8000,
  [string]$Model = "gemini-2.5-flash-lite"
)

$Root = Split-Path -Parent $PSScriptRoot

Push-Location $Root
try {
  python tools\rag_api_server.py `
    --host $HostAddress `
    --port $Port `
    --model $Model `
    --api-key-env GEMINI_API_KEY
}
finally {
  Pop-Location
}
