param(
  [Parameter(Mandatory = $true)]
  [string]$Version,
  [switch]$Tag,
  [switch]$SkipTests,
  [switch]$DryRun
)

$args = @("scripts/release.py", $Version)
if ($Tag) { $args += "--tag" }
if ($SkipTests) { $args += "--skip-tests" }
if ($DryRun) { $args += "--dry-run" }

python @args
