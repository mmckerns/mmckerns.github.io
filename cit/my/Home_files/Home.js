// Created by iWeb 3.0.3 local-build-20111008

setTransparentGifURL('Media/transparent.gif');function applyEffects()
{var registry=IWCreateEffectRegistry();registry.registerEffects({stroke_1:new IWPhotoFrame([IWCreateImage('Home_files/Creme_sidebar_frame_01.png'),IWCreateImage('Home_files/Creme_sidebar_frame_02.png'),IWCreateImage('Home_files/Creme_sidebar_frame_03.png'),IWCreateImage('Home_files/Creme_sidebar_frame_06.png'),IWCreateImage('Home_files/Creme_sidebar_frame_09.png'),IWCreateImage('Home_files/Creme_sidebar_frame_08.png'),IWCreateImage('Home_files/Creme_sidebar_frame_07.png'),IWCreateImage('Home_files/Creme_sidebar_frame_04.png')],null,2,1.000000,0.000000,0.000000,0.000000,0.000000,10.000000,16.000000,10.000000,20.000000,523.000000,173.000000,523.000000,173.000000,null,null,null,0.100000),stroke_0:new IWPhotoFrame([IWCreateImage('Home_files/Creme_frame2_01.png'),IWCreateImage('Home_files/Creme_frame2_02.png'),IWCreateImage('Home_files/Creme_frame2_04.png'),IWCreateImage('Home_files/Creme_frame2_08.png'),IWCreateImage('Home_files/Creme_frame2_09.png'),IWCreateImage('Home_files/Creme_frame2_14.png'),IWCreateImage('Home_files/Creme_frame2_13.png'),IWCreateImage('Home_files/Creme_frame2_05.png')],null,0,1.000000,3.000000,3.000000,3.000000,3.000000,23.000000,23.000000,23.000000,23.000000,20.000000,19.000000,20.000000,19.000000,null,null,null,0.400000)});registry.applyEffects();}
function hostedOnDM()
{return false;}
function onPageLoad()
{loadMozillaCSS('Home_files/HomeMoz.css')
adjustLineHeightIfTooBig('id1');adjustFontSizeIfTooBig('id1');adjustLineHeightIfTooBig('id2');adjustFontSizeIfTooBig('id2');detectBrowser();fixAllIEPNGs('Media/transparent.gif');Widget.onload();fixupAllIEPNGBGs();applyEffects()}
function onPageUnload()
{Widget.onunload();}
