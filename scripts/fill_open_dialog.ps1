param([string]$Path)

$sig = @"
using System;
using System.Runtime.InteropServices;
public class Dlg {
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr FindWindow(string cls, string title);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr FindWindowEx(IntPtr parent, IntPtr after, string cls, string title);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern bool SetWindowText(IntPtr h, string text);
  [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint msg, IntPtr w, IntPtr l);
}
"@
Add-Type -TypeDefinition $sig

$dlg = [Dlg]::FindWindow('#32770', '打开')
if ($dlg -eq [IntPtr]::Zero) {
  Write-Output "dialog not found"
  exit 1
}
$edit = [Dlg]::FindWindowEx($dlg, [IntPtr]::Zero, 'Edit', $null)
if ($edit -eq [IntPtr]::Zero) {
  Write-Output "edit not found"
  exit 2
}
[void][Dlg]::SetWindowText($edit, $Path)
Start-Sleep -Milliseconds 300
[void][Dlg]::PostMessage($dlg, 0x0111, [IntPtr]1, [IntPtr]::Zero)
Write-Output "ok"
