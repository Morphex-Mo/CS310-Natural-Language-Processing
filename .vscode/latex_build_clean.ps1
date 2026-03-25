param(
    [Parameter(Mandatory = $true)]
    [string]$FileDir,
    [Parameter(Mandatory = $true)]
    [string]$FileBaseName,
    [Parameter(Mandatory = $true)]
    [string]$FileBaseNameNoExt
)

$xelatex = "C:/Users/mxm/AppData/Local/Programs/MiKTeX/miktex/bin/x64/xelatex.exe"

Push-Location $FileDir
try {
    & $xelatex -synctex=0 -interaction=nonstopmode -file-line-error $FileBaseName
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $extensions = @(
        'aux',
        'log',
        'out',
        'toc',
        'nav',
        'snm',
        'fls',
        'fdb_latexmk',
        'xdv',
        'synctex.gz'
    )

    foreach ($ext in $extensions) {
        $path = Join-Path $FileDir ("$FileBaseNameNoExt.$ext")
        if (Test-Path $path) {
            Remove-Item $path -Force -ErrorAction SilentlyContinue
        }
    }
}
finally {
    Pop-Location
}
