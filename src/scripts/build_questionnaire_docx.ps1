param(
  [Parameter(Mandatory = $false)]
  [string]$Tag = "",

  [Parameter(Mandatory = $false)]
  [string]$InputMarkdown = "deliverables/20260206_aomori_survey_questionnaire_v1.md",

  [Parameter(Mandatory = $false)]
  [string]$OutputName = "aomori_survey_questionnaire_distribute.docx",

  [Parameter(Mandatory = $false)]
  [switch]$CopyToDeliverables,

  [Parameter(Mandatory = $false)]
  [string]$DeliverablesName = "202603_aomori_survey_questionnaire_distribute.docx"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")

function Resolve-PandocPath {
  $cmd = Get-Command pandoc -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source -and (Test-Path $cmd.Source)) {
    return $cmd.Source
  }

  $candidates = @(
    "C:\\Program Files\\Pandoc\\pandoc.exe",
    "C:\\Program Files (x86)\\Pandoc\\pandoc.exe"
  )

  if ($env:LOCALAPPDATA) {
    $wingetGlob = Join-Path $env:LOCALAPPDATA "Microsoft\\WinGet\\Packages\\JohnMacFarlane.Pandoc*\\pandoc-*\\pandoc.exe"
    $candidates += (Get-ChildItem -Path $wingetGlob -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
  }

  $candidates = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique
  if ($candidates -and $candidates.Count -gt 0) {
    function Parse-PandocVersion([string]$path) {
      $m = [regex]::Match($path, "pandoc-(\d+\.\d+\.\d+)", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
      if ($m.Success) {
        try { return [version]$m.Groups[1].Value } catch { return [version]"0.0.0" }
      }
      return [version]"0.0.0"
    }
    return ($candidates | Sort-Object { Parse-PandocVersion $_ } -Descending | Select-Object -First 1)
  }

  throw (
    "pandoc.exe が見つかりません。`n" +
    "インストール後に再実行してください。例: winget install --id JohnMacFarlane.Pandoc -e"
  )
}

if (-not $Tag) {
  $Tag = (Get-Date).ToString("yyyyMMdd_HHmmss") + "_questionnaire_docx"
}

$outDir = Join-Path $repoRoot ("outputs\\runs\\{0}" -f $Tag)
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$inputAbs = Join-Path $repoRoot $InputMarkdown
if (-not (Test-Path $inputAbs)) {
  throw ("InputMarkdown が見つかりません: {0}" -f $inputAbs)
}

$pandoc = Resolve-PandocPath
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  throw "python が見つかりません。Python 3.x をPATHに追加してください。"
}

$renderScript = Join-Path $repoRoot "src\\scripts\\render_questionnaire_distribution_md.py"
$refDocScript = Join-Path $repoRoot "src\\scripts\\make_questionnaire_reference_docx.py"
if (-not (Test-Path $renderScript)) { throw ("script not found: {0}" -f $renderScript) }
if (-not (Test-Path $refDocScript)) { throw ("script not found: {0}" -f $refDocScript) }

$intermediateMd = Join-Path $outDir "questionnaire_distribution.md"
$layoutReport = Join-Path $outDir "questionnaire_layout_report.json"
$referenceDoc = Join-Path $outDir "reference.docx"
$outputDocx = Join-Path $outDir $OutputName

Set-Location $repoRoot

Write-Host ("Rendering distribution markdown: {0}" -f $intermediateMd)
python $renderScript @(
  "--input", $inputAbs,
  "--output", $intermediateMd,
  "--report", $layoutReport
)

Write-Host ("Generating reference docx: {0}" -f $referenceDoc)
python $refDocScript @(
  "--pandoc", $pandoc,
  "--output", $referenceDoc
)

$pandocArgs = @(
  $intermediateMd,
  "--from", "markdown+pipe_tables",
  "--to", "docx",
  "--output", $outputDocx,
  "--reference-doc", $referenceDoc
)

$cmdTxt = Join-Path $outDir "pandoc_command.txt"
$quotedArgs = $pandocArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }
$cmdLine = '"' + ($pandoc -replace '"', '\"') + '"' + ' ' + ($quotedArgs -join ' ')
Set-Content -Path $cmdTxt -Value $cmdLine

Write-Host ("Running Pandoc -> {0}" -f $outputDocx)
& $pandoc @pandocArgs
if ($LASTEXITCODE -ne 0) {
  throw ("Pandoc failed (exit={0}). See: {1}" -f $LASTEXITCODE, $cmdTxt)
}

if ($CopyToDeliverables) {
  $deliverablesDir = Join-Path $repoRoot "deliverables"
  New-Item -ItemType Directory -Path $deliverablesDir -Force | Out-Null
  $finalPath = Join-Path $deliverablesDir $DeliverablesName
  Copy-Item -Path $outputDocx -Destination $finalPath -Force
  Write-Host ("Copied deliverable: {0}" -f $finalPath)
}

Write-Host ("Done: {0}" -f $outputDocx)
