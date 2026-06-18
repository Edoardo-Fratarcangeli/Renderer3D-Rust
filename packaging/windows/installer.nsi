; ─────────────────────────────────────────────────────────────────────────────
;  Rust 3D Renderer — modern multilingual Windows installer (NSIS / MUI2)
;
;  Build:
;    makensis -DAPP_VERSION=1.1.0 packaging\windows\installer.nsi
;
;  Overridable defines (passed with -D from CI):
;    APP_VERSION   product version            (default 0.0.0)
;    SRC_DIR       folder holding the .exe     (default target\release)
;    ICON_DIR      folder holding icons        (default packaging\icons)
;    OUT_FILE      output installer path       (default dist\...-setup.exe)
;
;  Pages (all localized into EN / IT / ES / FR / DE):
;    [language selector] → Welcome → "What is this software?" → Components
;    → Install directory → Install → Finish (with "launch app")
; ─────────────────────────────────────────────────────────────────────────────

Unicode true
ManifestDPIAware true
SetCompressor /SOLID lzma

!include "MUI2.nsh"
!include "nsDialogs.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"

; ── Product metadata ─────────────────────────────────────────────────────────
!ifndef APP_VERSION
  !define APP_VERSION "0.0.0"
!endif
!define APP_NAME      "Rust 3D Renderer"
!define APP_PUBLISHER "Edoardo Fratarcangeli"
!define APP_EXE       "rendering_3d.exe"
!define APP_URL       "https://github.com/Edoardo-Fratarcangeli/Renderer3D-Rust"
!define APP_REGKEY    "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

!ifndef SRC_DIR
  !define SRC_DIR "target\release"
!endif
!ifndef ICON_DIR
  !define ICON_DIR "packaging\icons"
!endif
!ifndef OUT_FILE
  !define OUT_FILE "dist\Rust-3D-Renderer-${APP_VERSION}-x64-setup.exe"
!endif

Name "${APP_NAME} ${APP_VERSION}"
OutFile "${OUT_FILE}"
InstallDir "$PROGRAMFILES64\${APP_NAME}"
InstallDirRegKey HKLM "Software\${APP_NAME}" "InstallDir"
RequestExecutionLevel admin

VIProductVersion "${APP_VERSION}.0"
VIAddVersionKey "ProductName"     "${APP_NAME}"
VIAddVersionKey "FileDescription" "High performance 3D renderer (WGPU + egui)"
VIAddVersionKey "CompanyName"     "${APP_PUBLISHER}"
VIAddVersionKey "LegalCopyright"  "Copyright (C) 2026 ${APP_PUBLISHER}"
VIAddVersionKey "FileVersion"     "${APP_VERSION}"
VIAddVersionKey "ProductVersion"  "${APP_VERSION}"

; ── MUI appearance ───────────────────────────────────────────────────────────
!define MUI_ICON   "${ICON_DIR}\icon.ico"
!define MUI_UNICON "${ICON_DIR}\icon.ico"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP        "packaging\windows\header.bmp"
!define MUI_WELCOMEFINISHPAGE_BITMAP  "packaging\windows\sidebar.bmp"
!define MUI_ABORTWARNING

; Remember the chosen installer language for next time / uninstall.
!define MUI_LANGDLL_REGISTRY_ROOT      "HKLM"
!define MUI_LANGDLL_REGISTRY_KEY       "Software\${APP_NAME}"
!define MUI_LANGDLL_REGISTRY_VALUENAME "Installer Language"

; ── Pages ────────────────────────────────────────────────────────────────────
!define MUI_WELCOMEPAGE_TITLE "$(WELCOME_TITLE)"
!define MUI_WELCOMEPAGE_TEXT  "$(WELCOME_TEXT)"
!insertmacro MUI_PAGE_WELCOME

Page custom AboutPageCreate                       ; "What is this software?"

!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES

!define MUI_FINISHPAGE_RUN       "$INSTDIR\${APP_EXE}"
!define MUI_FINISHPAGE_RUN_TEXT  "$(FINISH_RUN)"
!define MUI_FINISHPAGE_LINK      "$(FINISH_LINK)"
!define MUI_FINISHPAGE_LINK_LOCATION "${APP_URL}"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; ── Languages (order = order shown in the selector) ──────────────────────────
!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "Italian"
!insertmacro MUI_LANGUAGE "Spanish"
!insertmacro MUI_LANGUAGE "French"
!insertmacro MUI_LANGUAGE "German"

; Localized strings live in one .nsh per language (must be included AFTER the
; MUI_LANGUAGE lines so ${LANG_*} ids exist).
!include "packaging\windows\strings\English.nsh"
!include "packaging\windows\strings\Italian.nsh"
!include "packaging\windows\strings\Spanish.nsh"
!include "packaging\windows\strings\French.nsh"
!include "packaging\windows\strings\German.nsh"

; ── Custom "About / What is this software?" page ─────────────────────────────
Var AboutLabel

Function AboutPageCreate
  !insertmacro MUI_HEADER_TEXT "$(ABOUT_HEADER)" "$(ABOUT_SUBHEADER)"
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 100% "$(ABOUT_TEXT)"
  Pop $AboutLabel
  nsDialogs::Show
FunctionEnd

; ── Install sections ─────────────────────────────────────────────────────────
Section "$(SEC_CORE)" SecCore
  SectionIn RO
  SetOutPath "$INSTDIR"
  File "${SRC_DIR}\${APP_EXE}"

  WriteRegStr HKLM "Software\${APP_NAME}" "InstallDir" "$INSTDIR"
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Add/Remove Programs entry.
  WriteRegStr   HKLM "${APP_REGKEY}" "DisplayName"     "${APP_NAME}"
  WriteRegStr   HKLM "${APP_REGKEY}" "DisplayVersion"  "${APP_VERSION}"
  WriteRegStr   HKLM "${APP_REGKEY}" "Publisher"       "${APP_PUBLISHER}"
  WriteRegStr   HKLM "${APP_REGKEY}" "DisplayIcon"     "$INSTDIR\${APP_EXE}"
  WriteRegStr   HKLM "${APP_REGKEY}" "URLInfoAbout"    "${APP_URL}"
  WriteRegStr   HKLM "${APP_REGKEY}" "UninstallString" "$INSTDIR\Uninstall.exe"
  WriteRegStr   HKLM "${APP_REGKEY}" "QuietUninstallString" "$INSTDIR\Uninstall.exe /S"
  WriteRegDWORD HKLM "${APP_REGKEY}" "NoModify" 1
  WriteRegDWORD HKLM "${APP_REGKEY}" "NoRepair" 1

  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${APP_REGKEY}" "EstimatedSize" "$0"
SectionEnd

Section "$(SEC_STARTMENU)" SecStartMenu
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut  "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
  CreateShortcut  "$SMPROGRAMS\${APP_NAME}\$(SHORTCUT_UNINSTALL).lnk" "$INSTDIR\Uninstall.exe"
SectionEnd

Section "$(SEC_DESKTOP)" SecDesktop
  CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
SectionEnd

; Component descriptions (shown on the Components page).
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCore}      "$(DESC_SecCore)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SecStartMenu} "$(DESC_SecStartMenu)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop}   "$(DESC_SecDesktop)"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ── Uninstall ────────────────────────────────────────────────────────────────
Section "Uninstall"
  Delete "$INSTDIR\${APP_EXE}"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir  "$INSTDIR"

  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\$(SHORTCUT_UNINSTALL).lnk"
  RMDir  "$SMPROGRAMS\${APP_NAME}"
  Delete "$DESKTOP\${APP_NAME}.lnk"

  DeleteRegKey HKLM "${APP_REGKEY}"
  DeleteRegKey HKLM "Software\${APP_NAME}"
SectionEnd

; ── Init (language selection) ────────────────────────────────────────────────
Function .onInit
  !insertmacro MUI_LANGDLL_DISPLAY
FunctionEnd

Function un.onInit
  !insertmacro MUI_UNGETLANGUAGE
FunctionEnd
