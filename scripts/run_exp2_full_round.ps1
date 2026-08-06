# Full-coverage Exp2 teacher round runner v2 (2026-08-06, after max_tokens fix).
# Order: Aegis validation calibration -> DNA -> OR -> Fraud-R1 -> Aegis prompt.
$ErrorActionPreference = "Continue"
$root = "C:\Users\18201\Desktop\FraudDistill"
$log = Join-Path $root "outputs\exp2_full_run_v2.log"
Set-Location $root
function Run-Source([string]$src, [string]$mode, [string]$extra) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $log -Value "[$stamp] START $src $mode $extra"
    if ($extra -ne "") {
        & python scripts\run_exp2_teacher.py $extra 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
    } elseif ($mode -ne "") {
        & python scripts\run_exp2_teacher.py --benchmark $src --mode $mode 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
    } else {
        & python scripts\run_exp2_teacher.py --benchmark $src 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
    }
    $code = $LASTEXITCODE
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $log -Value "[$stamp] DONE $src $mode exit=$code"
}
Run-Source "aegis2" "" "--calib-aegis 300"
Run-Source "do_not_answer" "" ""
Run-Source "orbench" "" ""
Run-Source "fraudr1" "" ""
Run-Source "aegis2" "prompt" ""
Add-Content -Path $log -Value ("[ALL DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "]")
