param(
  [Parameter(Mandatory = $false)]
  [string]$Tag = "",

  [Parameter(Mandatory = $false)]
  [string]$InputMarkdown = "paper_en.md",

  [Parameter(Mandatory = $false)]
  [string]$OutputName = "paper_en.docx",

  [Parameter(Mandatory = $false)]
  [string]$BibPath = "",

  [Parameter(Mandatory = $false)]
  [string]$CslPath = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")

if (-not $Tag) {
  $Tag = (Get-Date).ToString("yyyyMMdd_HHmmss") + "_paper_docx"
}

$outDir = Join-Path $repoRoot ("outputs\\runs\\{0}" -f $Tag)
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$pandocCmd = Get-Command pandoc -ErrorAction SilentlyContinue
if (-not $pandocCmd) {
  throw "pandoc.exe が見つかりません。PATHにpandocを追加してください。"
}
$pandoc = $pandocCmd.Source

$inputAbs = Join-Path $repoRoot $InputMarkdown
if (-not (Test-Path $inputAbs)) {
  throw ("InputMarkdown が見つかりません: {0}" -f $inputAbs)
}

$outputDocx = Join-Path $outDir $OutputName
$pandocArgs = @(
  $inputAbs,
  "--from", "markdown",
  "--to", "docx",
  "--output", $outputDocx
)

if ($BibPath) {
  $bibAbs = Join-Path $repoRoot $BibPath
  if (-not (Test-Path $bibAbs)) {
    throw ("BibPath が見つかりません: {0}" -f $bibAbs)
  }
  $pandocArgs += @("--citeproc", "--bibliography", $bibAbs)
}

if ($CslPath) {
  $cslAbs = Join-Path $repoRoot $CslPath
  if (-not (Test-Path $cslAbs)) {
    throw ("CslPath が見つかりません: {0}" -f $cslAbs)
  }
  $pandocArgs += @("--csl", $cslAbs)
}

$cmdTxt = Join-Path $outDir "pandoc_command.txt"
$quotedArgs = $pandocArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }
$cmdLine = '"' + ($pandoc -replace '"', '\"') + '"' + ' ' + ($quotedArgs -join ' ')
Set-Content -Path $cmdTxt -Value $cmdLine

Write-Host ("Running Pandoc -> {0}" -f $outputDocx)
& $pandoc @pandocArgs
if ($LASTEXITCODE -ne 0) {
  throw ("Pandoc failed (exit={0}). See: {1}" -f $LASTEXITCODE, $cmdTxt)
}

Write-Host ("Done: {0}" -f $outputDocx)

