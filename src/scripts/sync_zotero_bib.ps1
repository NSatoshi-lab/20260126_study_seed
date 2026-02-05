param(
  [Parameter(Mandatory = $false)]
  [string]$BibName = "aomori_survey.bib",

  [Parameter(Mandatory = $false)]
  [string]$SourcePath,

  [Parameter(Mandatory = $false)]
  [string]$DestinationPath,

  [switch]$DryRun
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")

if (-not $DestinationPath) {
  $DestinationPath = Join-Path $repoRoot ("refs\\bib\\{0}" -f $BibName)
}

if (-not $SourcePath) {
  $oneDriveRoot = @(
    $env:OneDriveCommercial,
    $env:OneDriveConsumer,
    $env:OneDrive
  ) | Where-Object { $_ } | Select-Object -First 1

  if (-not $oneDriveRoot) {
    throw "OneDriveの環境変数（OneDrive/OneDriveCommercial/OneDriveConsumer）が見つかりません。-SourcePath で直接指定してください。"
  }

  $SourcePath = Join-Path $oneDriveRoot ("ZoteroLibrary\\{0}" -f $BibName)
}

if (-not (Test-Path $SourcePath)) {
  throw ("SourcePath が見つかりません: {0}" -f $SourcePath)
}

$destDir = Split-Path -Parent $DestinationPath
if (-not (Test-Path $destDir)) {
  if ($DryRun) {
    Write-Host ("[DRYRUN] mkdir {0}" -f $destDir)
  } else {
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
  }
}

if ($DryRun) {
  Write-Host ("[DRYRUN] copy {0} -> {1}" -f $SourcePath, $DestinationPath)
} else {
  Copy-Item -Path $SourcePath -Destination $DestinationPath -Force
  Write-Host ("synced: {0} -> {1}" -f $SourcePath, $DestinationPath)
}
