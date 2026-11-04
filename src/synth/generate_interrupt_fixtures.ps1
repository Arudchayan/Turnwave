param([Parameter(Mandatory=$true)][string]$OutDir)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutDir) { throw 'Output directory already exists; choose a new one.' }
New-Item -ItemType Directory -Path $OutDir | Out-Null
Add-Type -AssemblyName System.Speech
$items = @(@('ack','mm hmm'), @('yes','yes'), @('stop','stop'), @('wait','wait a moment'))
foreach ($voice in @('Microsoft David Desktop','Microsoft Zira Desktop')) {
    foreach ($item in $items) {
        $synth = [System.Speech.Synthesis.SpeechSynthesizer]::new()
        try {
            $synth.SelectVoice($voice)
            $label = if ($voice -like '*David*') { 'david' } else { 'zira' }
            $file = Join-Path $OutDir ($label + '-' + $item[0] + '.wav')
            $format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
                16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
                [System.Speech.AudioFormat.AudioChannel]::Mono)
            $synth.SetOutputToWaveFile($file, $format)
            $synth.Speak($item[1])
        } finally { $synth.Dispose() }
    }
}
