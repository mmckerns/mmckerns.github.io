// Created by iWeb 3.0.3 local-build-20110628

function writeMovie1()
{detectBrowser();if(windowsInternetExplorer)
{document.write('<object id="id1" classid="clsid:02BF25D5-8C17-4B23-BC80-D3488ABDDC6B" codebase="http://www.apple.com/qtactivex/qtplugin.cab" width="320" height="256" style="height: 256px; left: 80px; position: absolute; top: 38px; width: 320px; z-index: 1; "><param name="src" value="../../Media/AA048452_a-1.mp4" /><param name="controller" value="true" /><param name="autoplay" value="false" /><param name="scale" value="tofit" /><param name="volume" value="100" /><param name="loop" value="false" /></object>');}
else if(isiPhone)
{document.write('<object id="id1" type="video/quicktime" width="320" height="256" style="height: 256px; left: 80px; position: absolute; top: 38px; width: 320px; z-index: 1; "><param name="src" value="Demos_files/AA048452_a.jpg"/><param name="target" value="myself"/><param name="href" value="../../../Media/AA048452_a-1.mp4"/><param name="controller" value="true"/><param name="scale" value="tofit"/></object>');}
else
{document.write('<object id="id1" type="video/quicktime" width="320" height="256" data="../../Media/AA048452_a-1.mp4" style="height: 256px; left: 80px; position: absolute; top: 38px; width: 320px; z-index: 1; "><param name="src" value="../../Media/AA048452_a-1.mp4"/><param name="controller" value="true"/><param name="autoplay" value="false"/><param name="scale" value="tofit"/><param name="volume" value="100"/><param name="loop" value="false"/></object>');}}
setTransparentGifURL('../../Media/transparent.gif');function applyEffects()
{var registry=IWCreateEffectRegistry();registry.registerEffects({shadow_0:new IWShadow({blurRadius:3,offset:new IWPoint(0.1736,0.9848),color:'#817b67',opacity:0.700000})});registry.applyEffects();}
function hostedOnDM()
{return false;}
function onPageLoad()
{loadMozillaCSS('Demos_files/DemosMoz.css')
adjustLineHeightIfTooBig('id2');adjustFontSizeIfTooBig('id2');adjustLineHeightIfTooBig('id3');adjustFontSizeIfTooBig('id3');fixAllIEPNGs('../../Media/transparent.gif');Widget.onload();applyEffects()}
function onPageUnload()
{Widget.onunload();}
