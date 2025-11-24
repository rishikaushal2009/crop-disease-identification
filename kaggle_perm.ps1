$kagPath = "$env:USERPROFILE\.kaggle"
       #New-Item -ItemType Directory -Force -Path $kagPath
       #Copy-Item .\kaggle.json $kagPath
       icacls "$kagPath\kaggle.json" /inheritance:r
       icacls "$kagPath\kaggle.json" /grant:r "$($env:USERNAME):(R)" "Administrators:(R)"