if (Test-Path -Path "./WindowsNoEditor") {
    "release already exists!"
    Pause
    Exit
}

# Download latest dotnet/codeformatter release from github
$repo = "dictor/endurance-environment"
$file = "WindowsNoEditor.zip"

$releases = "https://api.github.com/repos/$repo/releases"

Write-Host Determining latest release
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$tag = (Invoke-WebRequest -Uri $releases -UseBasicParsing | ConvertFrom-Json)[0].tag_name

$download = "https://github.com/$repo/releases/download/$tag/$file"
$name = $file.Split(".")[0]
$zip = "$name-$tag.zip"
$dir = "$name-$tag"

Write-Host Dowloading latest release

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest $download -OutFile $zip

Write-Host Extracting release files
Expand-Archive $zip -Force

# Moving from temp dir to target dir
Rename-Item -Path $name-$tag -NewName $name -Force

# Removing temp files
Remove-Item $zip -Force