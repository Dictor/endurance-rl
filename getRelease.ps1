# origin source from https://gist.github.com/f3l3gy/0e89dde158dde024959e36e915abf6bd

if (Test-Path -Path "./WindowsNoEditor") {
    Write-Host "release already exists! exisiting release will be deleted and replaced to new. type 'y' for continue" -BackgroundColor 'White' -ForegroundColor 'Red'
    $ans = Read-Host
    if ($ans -ne "y") {
        Exit
    }
}

# Download latest dotnet/codeformatter release from github
$repo = "dictor/endurance-environment"
$file = "WindowsNoEditor.zip"
$name = $file.Split(".")[0]
Write-Host "Remove existing"
Remove-Item -Path $name -Recurse

$releases = "https://api.github.com/repos/$repo/releases"

Write-Host Determining latest release
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$tag = (Invoke-WebRequest -Uri $releases -UseBasicParsing | ConvertFrom-Json)[0].tag_name

$download = "https://github.com/$repo/releases/download/$tag/$file"
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