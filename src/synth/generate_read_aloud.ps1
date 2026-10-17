param([Parameter(Mandatory=$true)][string]$Experiment)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$protocol = Get-Content -LiteralPath (Join-Path $Experiment 'protocol.json') -Raw -Encoding utf8 | ConvertFrom-Json
$audioFolder = Join-Path $Experiment 'fixtures'
New-Item -ItemType Directory -Path $audioFolder -Force | Out-Null
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono)
try {
    foreach ($voice in $protocol.voices) {
        $synth.SelectVoice($voice.name)
        foreach ($case in $protocol.cases) {
            $file = Join-Path $audioFolder ($voice.id + '-' + $case.id + '.wav')
            if (Test-Path -LiteralPath $file) { throw "Refusing to overwrite existing fixture: $file" }
            $synth.SetOutputToWaveFile($file,$format)
            $synth.SpeakSsml($case.ssml)
            $synth.SetOutputToNull()
            $row = @{ts=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000;voice=$voice.id;case=$case.id;file=$file}
            $row | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $Experiment 'fixture-progress.jsonl') -Encoding utf8
            Write-Output ($voice.id + ' ' + $case.id)
        }
    }
} finally {
    $synth.Dispose()
}
