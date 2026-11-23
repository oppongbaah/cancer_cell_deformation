; Inno Setup script for the celldform installer. Build with:  iscc /DAppVersion=0.1.0 celldform.iss
; Installs for the current user only (no administrator needed), offers a desktop shortcut (ticked by default) and a
; Start Menu entry, and uninstalls cleanly. It installs the program only: no models, no data, no telemetry.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define AppName "celldform"

[Setup]
AppId={{E183070B-3812-4471-88CB-522CC78F23AD}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
; a neutral publisher: nothing about whoever built it
AppPublisher={#AppName}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist\installer
OutputBaseFilename=celldform-setup-{#AppVersion}
SetupIconFile=..\..\celldform\demo\assets\celldform.ico
UninstallDisplayIcon={app}\celldform.exe
UninstallDisplayName={#AppName}
LicenseFile=..\..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; the installer asks nothing about the user and writes nothing outside {app}, the shortcuts and the uninstall entry
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "startmenuicon"; Description: "Add to the &Start Menu"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\..\dist\celldform\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\celldform.exe"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\celldform.exe"; WorkingDir: "{app}"; Tasks: startmenuicon

[Run]
Filename: "{app}\celldform.exe"; Description: "Start {#AppName} now"; Flags: nowait postinstall skipifsilent

[Code]
// Uninstalling removes the program. The user's own settings (which files were loaded) and results are kept unless they ask.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if MsgBox('Also delete celldform''s saved settings (the list of model files you loaded)?' + #13#10 +
              'Your results in Documents\celldform are never deleted.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      DelTree(ExpandConstant('{userappdata}\celldform'), True, True, True);
end;
