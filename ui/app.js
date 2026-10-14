import {encodePrompt} from './inference.js';
const $=id=>document.getElementById(id);
const names={trained:'内部Weight学習',fixed:'固定Reservoir',no_memory:'記憶なし',gru:'小GRU'};
let bundle=null,variant='trained',busy=false,workers=[],runNumber=0,lastResult=null,latestVisualization=null;
const examples={citizen:'First Citizen:\n',romeo:'ROMEO:\n',tobe:'To be, or not to be:\n'};

function notify(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(notify.timer);notify.timer=setTimeout(()=>$('toast').hidden=true,2300);}
function recordFor(type=variant){return bundle?.models.find(m=>m.variant===type&&m.seed===Number($('checkpoint-seed').value));}
function updateSelection(){
  if(lastResult&&!busy){
    lastResult=null;
    $('output-primary').hidden=true;$('output-comparison').hidden=true;
    $('empty-primary').hidden=false;$('empty-comparison').hidden=false;
    $('copy-primary').disabled=true;$('download').disabled=true;
    $('run-status').textContent='準備完了';$('primary-speed').textContent='— char/s';
    $('comparison-speed').textContent='— char/s';$('comparison-progress').textContent='比較待ち';
    $('primary-progress').textContent=`0 / ${$('characters').value}文字`;
    $('progress-bar').style.width='0%';$('next-character').textContent='次の文字候補 —';
  }
  const seed=$('checkpoint-seed').value,comparison=variant==='fixed'?'trained':'fixed';
  $('output-label').textContent=names[variant];$('output-seed').textContent=`SEED ${seed}`;
  $('comparison-label').textContent=names[comparison];$('comparison-seed').textContent=`SEED ${seed}`;
  $('comparison-panel').hidden=!$('compare').checked;
  $('compare').parentElement.lastElementChild.textContent=variant==='fixed'?'内部学習版と並べて比較':'固定版と並べて比較';
  $('checkpoint-name').textContent=`${variant} · seed ${seed}`;
  const record=recordFor();if(record)$('checkpoint-update').textContent=`best step ${record.best_update}`;
  for(const button of document.querySelectorAll('[data-variant]')){
    const active=button.dataset.variant===variant;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));
  }
  for(const card of document.querySelectorAll('[data-metric]'))card.classList.toggle('selected',card.dataset.metric===variant);
  $('state-heading').textContent=variant==='gru'?'GRUのデジタル状態':'波動状態をのぞく';
  $('state-description').textContent=variant==='gru'?'小GRUはデジタル比較モデルのため、YIGの波動状態はありません。':'生成中の空間32点の振幅。Readoutへ渡るのはアンテナ観測だけです。';
  latestVisualization=null;drawState(null);
}
function updateValues(){
  $('prompt-count').textContent=`${[...$('prompt').value].length}文字`;
  $('characters-value').textContent=$('characters').value;
  $('temperature-value').textContent=Number($('temperature').value).toFixed(2);
  if(!busy&&!lastResult)$('primary-progress').textContent=`0 / ${$('characters').value}文字`;
}
function setBusy(value){
  busy=value;
  for(const control of document.querySelectorAll('#generate-form input,#generate-form textarea,#generate-form select,#generate-form button'))control.disabled=value||!bundle;
  $('generate-button').querySelector('span').textContent=value?'生成しています…':'続きを生成';
  $('start-example').disabled=value||!bundle;$('stop').hidden=!value;
  $('run-status').classList.toggle('running',value);
  $('output-primary').querySelector('.text-caret').hidden=!value;
}
function prepareOutput(target,prompt){
  const primary=target==='primary',pre=$(primary?'output-primary':'output-comparison');
  $(primary?'empty-primary':'empty-comparison').hidden=true;pre.hidden=false;
  pre.querySelector('.prompt-prefix').textContent=prompt;
  pre.querySelector('.generated-text').textContent='';
}
function drawState(visualization){
  const canvas=$('state-canvas'),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
  if(!rect.width)return;
  canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);
  const context=canvas.getContext('2d');context.scale(ratio,ratio);
  const width=rect.width,height=rect.height;context.clearRect(0,0,width,height);
  context.strokeStyle='#dbe5d0';context.lineWidth=1;context.beginPath();context.moveTo(0,height-5);context.lineTo(width,height-5);context.stroke();
  if(!visualization){
    context.fillStyle='#9ba98d';context.font='9px -apple-system,sans-serif';
    context.fillText(variant==='gru'?'Digital GRU · 波動状態は表示しません':'推論を始めると、計算した振幅が表示されます',0,height/2+4);
    $('state-legend').textContent=variant==='gru'?'DIGITAL BASELINE':'SPACE 0 → 800 μm';return;
  }
  const values=visualization.amplitude,maximum=Math.max(...values,1e-9),step=width/values.length;
  for(let i=0;i<values.length;i++) {
    const barHeight=values[i]/maximum*(height-13);
    context.fillStyle=i%2?'#7ba88d':'#609677';
    context.fillRect(i*step+2,height-5-barHeight,Math.max(2,step-5),barHeight);
  }
  $('state-legend').textContent=`SPACE 0 → 800 μm · MAX ${maximum.toExponential(2)}`;
}
function stopGeneration(){
  if(!busy)return;
  runNumber++;workers.forEach(worker=>worker.terminate());workers=[];
  setBusy(false);$('run-status').textContent='停止しました';
  if(lastResult){lastResult.status='stopped';$('copy-primary').disabled=!lastResult.primary.text.length;$('download').disabled=false;}
}
async function startGeneration(event){
  event?.preventDefault();if(busy||!bundle)return;
  $('form-error').hidden=true;
  const prompt=$('prompt').value,characters=Number($('characters').value),temperature=Number($('temperature').value),randomSeed=Number($('random-seed').value);
  try{encodePrompt(prompt,bundle.vocabulary);if(!Number.isInteger(randomSeed)||randomSeed<0||randomSeed>4294967295)throw new Error('生成Seedは0〜4294967295の整数にしてください。');}
  catch(error){$('form-error').textContent=error.message;$('form-error').hidden=false;return;}
  const options={prompt,characters,temperature,randomSeed,topK:Number($('top-k').value),greedy:$('greedy').checked};
  const primary=recordFor(),reference=recordFor(variant==='fixed'?'trained':'fixed');
  const selections=[{target:'primary',record:primary}];if($('compare').checked)selections.push({target:'comparison',record:reference});
  $('output-comparison').hidden=true;$('empty-comparison').hidden=false;
  $('empty-comparison').textContent=selections.length===2?'同じ条件で比較します。':'比較するには、もう一度「続きを生成」を押してください。';
  lastResult={created_at:new Date().toISOString(),status:'running',backend:'Browser CPU, actual exported checkpoint weights',options,
    primary:{model:primary.id,checkpoint_sha256:primary.checkpoint_sha256,text:'',generated:0},
    comparison:selections.length===2?{model:reference.id,checkpoint_sha256:reference.checkpoint_sha256,text:'',generated:0}:null};
  $('copy-primary').disabled=true;$('download').disabled=true;
  $('progress-bar').style.width='0%';$('primary-progress').textContent=`0 / ${characters}文字`;$('primary-speed').textContent='— char/s';
  $('comparison-progress').textContent=selections.length===2?'生成を準備中':'比較待ち';$('comparison-speed').textContent='— char/s';
  $('next-character').textContent='次の文字候補 —';
  latestVisualization=null;drawState(null);setBusy(true);$('run-status').textContent='生成中';
  let finished=0;const currentRun=++runNumber;
  function fail(message){
    if(currentRun!==runNumber)return;
    stopGeneration();
    if(lastResult){lastResult.status='error';lastResult.error=message;}
    $('form-error').textContent=message;$('form-error').hidden=false;$('run-status').textContent='生成エラー';
  }
  workers=[];
  try {for(const {target,record} of selections){
    prepareOutput(target,prompt);
    const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
    workers.push(worker);
    worker.onerror=()=>fail('推論workerを起動できませんでした。ページを再読み込みしてください。');
    worker.onmessage=({data})=>{
      if(currentRun!==runNumber)return;
      if(data.type==='ready'){worker.postMessage({type:'generate',modelId:record.id,...options});return;}
      if(data.type==='error'){fail(data.message);return;}
      if(data.type!=='progress'&&data.type!=='done')return;
      lastResult[target].text=data.text;lastResult[target].generated=data.generated;lastResult[target].elapsed_ms=data.elapsedMs;
      $(target==='primary'?'output-primary':'output-comparison').querySelector('.generated-text').textContent=data.text;
      $(target==='primary'?'primary-progress':'comparison-progress').textContent=`${data.generated} / ${characters}文字`;
      $(target==='primary'?'primary-speed':'comparison-speed').textContent=`${(data.generated/Math.max(.001,data.elapsedMs/1000)).toFixed(1)} char/s`;
      if(target==='primary'){
        $('progress-bar').style.width=`${data.generated/characters*100}%`;
        if(data.type==='progress'){
          latestVisualization=data.visualization;drawState(latestVisualization);
          $('next-character').textContent='次: '+data.next.slice(0,3).map(x=>`${x.character==='\n'?'↵':x.character===' '?'␣':x.character} ${(x.probability*100).toFixed(0)}%`).join(' · ');
        }
        $('copy-primary').disabled=false;$('download').disabled=false;
      }
      if(data.type==='done'){
        worker.terminate();finished++;
        if(finished===selections.length){workers=[];lastResult.status='completed';setBusy(false);$('run-status').textContent='生成完了';}
      }
    };
    worker.postMessage({type:'init'});
  }} catch {fail('このブラウザでは推論workerを起動できませんでした。別のブラウザでお試しください。');}
}

$('generate-form').addEventListener('submit',startGeneration);
$('start-example').addEventListener('click',startGeneration);
$('stop').addEventListener('click',stopGeneration);
for(const button of document.querySelectorAll('[data-variant]'))button.addEventListener('click',()=>{if(!busy){variant=button.dataset.variant;updateSelection();}});
for(const button of document.querySelectorAll('[data-prompt]'))button.addEventListener('click',()=>{$('prompt').value=examples[button.dataset.prompt];updateValues();});
for(const id of ['prompt','characters','temperature'])$(id).addEventListener('input',updateValues);
$('checkpoint-seed').addEventListener('change',updateSelection);
$('compare').addEventListener('change',()=>{$('comparison-panel').hidden=!$('compare').checked;});
$('copy-primary').addEventListener('click',async()=>{
  const text=lastResult.options.prompt+lastResult.primary.text;
  try{await navigator.clipboard.writeText(text);notify('生成結果をコピーしました');}
  catch{$('output-primary').focus();notify('コピーできませんでした。出力の文字を選択してください。');}
});
$('download').addEventListener('click',()=>{
  const blob=new Blob([JSON.stringify(lastResult,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob);
  const link=document.createElement('a');link.href=url;link.download=`spinlanguage-${lastResult.primary.model}-${lastResult.options.randomSeed}.json`;link.click();URL.revokeObjectURL(url);notify('結果と生成条件を保存しました');
});
document.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key==='Enter'){event.preventDefault();startGeneration();}});
window.addEventListener('resize',()=>drawState(latestVisualization));
updateValues();updateSelection();
try{
  const response=await fetch('./models.json');if(!response.ok)throw new Error('モデルファイルを読み込めませんでした。');
  bundle=await response.json();setBusy(false);updateSelection();$('run-status').textContent='準備完了';
  for(const card of document.querySelectorAll('[data-metric]'))card.querySelector('strong').textContent=bundle.aggregate[card.dataset.metric].ce_mean.toFixed(4);
}catch(error){$('generate-button').querySelector('span').textContent='読み込み失敗';$('run-status').textContent='読み込みエラー';$('form-error').textContent=error.message;$('form-error').hidden=false;}
