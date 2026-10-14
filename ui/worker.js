/** Worker isolates the actual recurrent simulation from the interactive UI. */
import {createRuntime,encodePrompt,seededRandom,sampleToken,samplingProbabilities} from './inference.js';
let bundle=null;
self.onmessage=async ({data})=>{
  try {
    if(data.type==='init') {
      const response=await fetch(new URL('./models.json',import.meta.url));
      if(!response.ok)throw new Error('モデルの読み込みに失敗しました。ページを再読み込みしてください。');
      bundle=await response.json();
      self.postMessage({type:'ready'});return;
    }
    if(data.type!=='generate' || !bundle)return;
    const started=performance.now(),tokens=encodePrompt(data.prompt,bundle.vocabulary);
    const record=bundle.models.find(m=>m.id===data.modelId);
    if(!record)throw new Error('指定したモデルが見つかりません。');
    const runtime=createRuntime(record),random=seededRandom(data.randomSeed);
    let logits;
    for(const token of tokens)logits=runtime.step(token);
    let text='';
    for(let position=0;position<data.characters;position++) {
      const token=sampleToken(logits,data,random);text+=bundle.vocabulary[token];
      logits=runtime.step(token);
      if((position+1)%8===0 || position===data.characters-1) {
        const probabilities=samplingProbabilities(logits,data);
        const next=Array.from(probabilities,(p,i)=>({character:bundle.vocabulary[i],probability:p})).sort((a,b)=>b.probability-a.probability).slice(0,5);
        self.postMessage({type:'progress',text,generated:position+1,elapsedMs:performance.now()-started,
          visualization:runtime.visualization(),next});
      }
    }
    self.postMessage({type:'done',text,generated:data.characters,elapsedMs:performance.now()-started});
  } catch(error) {self.postMessage({type:'error',message:error.message||'生成に失敗しました。'});}
};
