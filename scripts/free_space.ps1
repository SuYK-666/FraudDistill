$ErrorActionPreference = "SilentlyContinue"
Get-ChildItem "$env:TEMP" -Recurse -File | Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-1) } | Remove-Item -Force
$targets = "bge-reranker-base","xlm-roberta-base","multilingual-e5-small","paraphrase-multilingual-MiniLM","finbert-tone-chinese","bge-small-zh"
Get-ChildItem "C:\Users\18201\.cache\huggingface\hub" -Directory | ForEach-Object {
    foreach ($t in $targets) {
        if ($_.Name -like "*$t*") { Remove-Item -Recurse -Force $_.FullName }
    }
}
Get-PSDrive C | ForEach-Object { "C free: " + [math]::Round($_.Free/1GB,2) + " GB" }
