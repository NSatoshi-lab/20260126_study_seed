param(
  [Parameter(Mandatory = $false)]
  [string]$BridgeBaseUrl = "http://127.0.0.1:23119",

  [Parameter(Mandatory = $false)]
  [string]$EnvPath = ".env",

  [Parameter(Mandatory = $false)]
  [string]$CollectionName = "入浴統計"
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$envPathAbs = Resolve-Path (Join-Path $repoRoot $EnvPath) -ErrorAction SilentlyContinue
if (-not $envPathAbs) {
  throw (".env not found: {0}" -f (Join-Path $repoRoot $EnvPath))
}

$tokenLine = (Get-Content $envPathAbs | Where-Object { $_ -match '^\s*ZOTERO_BRIDGE_TOKEN\s*=' } | Select-Object -First 1)
if (-not $tokenLine) {
  throw ("ZOTERO_BRIDGE_TOKEN not found in {0}" -f $envPathAbs)
}

$token = ($tokenLine -split '=', 2)[1].Trim().Trim('"').Trim("'")
if (-not $token) {
  throw ("ZOTERO_BRIDGE_TOKEN is empty in {0}" -f $envPathAbs)
}

$pingUrl = ("{0}/codex/ping" -f $BridgeBaseUrl)
$importUrl = ("{0}/codex/import" -f $BridgeBaseUrl)

Write-Host ("ping: {0}" -f $pingUrl)
$ping = Invoke-RestMethod -Headers @{ "Zotero-Allowed-Request" = "1" } $pingUrl
$ping | ConvertTo-Json -Depth 5

$body = @{
  collection = $CollectionName
  item = @{
    itemType = "journalArticle"
    title = "Token Test"
  }
} | ConvertTo-Json -Compress

Write-Host ("import: {0}" -f $importUrl)
try {
  $resp = Invoke-RestMethod -Method Post -Uri $importUrl -Headers @{
    Authorization = ("Bearer {0}" -f $token)
    "Content-Type" = "application/json"
    "Zotero-Allowed-Request" = "1"
  } -Body $body
  $resp | ConvertTo-Json -Depth 10
} catch {
  $status = $null
  $detail = $null
  $http = $_.Exception.Response
  if ($http -and $http -is [System.Net.Http.HttpResponseMessage]) {
    $status = [int]$http.StatusCode
    try { $detail = $http.Content.ReadAsStringAsync().GetAwaiter().GetResult() } catch { }
  }
  if ($status) {
    Write-Error ("HTTP {0} for {1}" -f $status, $importUrl)
  } else {
    Write-Error ("Request failed for {0}" -f $importUrl)
  }
  if ($detail) {
    Write-Error $detail
  }
  throw
}
