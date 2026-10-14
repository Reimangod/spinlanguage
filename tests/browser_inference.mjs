import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRuntime,encodePrompt,softmax,seededRandom,sampleToken,samplingProbabilities} from '../ui/inference.js';
const bundle=JSON.parse(readFileSync(new URL('../ui/models.json',import.meta.url)));
const fixtures=JSON.parse(readFileSync(new URL('./browser_parity_fixtures.json',import.meta.url)));
const markup=readFileSync(new URL('../ui/index.html',import.meta.url),'utf8');
const ids=Array.from(markup.matchAll(/\bid="([^"]+)"/g),match=>match[1]);
assert.equal(new Set(ids).size,ids.length,'Duplicate HTML IDs can break UI controls');
let maxLogitError=0,maxStateError=0;
const flat=x=>Array.isArray(x)?x.flat(Infinity):Array.from(x);
for(const fixture of fixtures) {
  const record=bundle.models.find(m=>m.id===fixture.model_id),runtime=createRuntime(record);
  const tokens=encodePrompt(fixture.prompt,bundle.vocabulary);
  const expected=new Map(fixture.logits.map(x=>[x.position,x.values]));
  for(let position=0;position<tokens.length;position++) {
    const logits=runtime.step(tokens[position]);
    if(expected.has(position))for(let i=0;i<logits.length;i++)
      maxLogitError=Math.max(maxLogitError,Math.abs(logits[i]-expected.get(position)[i]));
  }
  const actual=flat(runtime.state()),reference=flat(fixture.state);
  assert.equal(actual.length,reference.length);
  for(let i=0;i<actual.length;i++)maxStateError=Math.max(maxStateError,Math.abs(actual[i]-reference[i]));
}
assert.ok(maxLogitError<.001,`Logit error ${maxLogitError}`);
assert.ok(maxStateError<.0001,`State error ${maxStateError}`);
assert.throws(()=>encodePrompt('こんにちは',bundle.vocabulary),/学習語彙/);
assert.throws(()=>encodePrompt('',bundle.vocabulary));
assert.ok(Math.abs(softmax([2,1,-1]).reduce((a,b)=>a+b,0)-1)<1e-12);
assert.equal(sampleToken([1,3,2],{greedy:true}),1);
const a=seededRandom(42),b=seededRandom(42);
assert.deepEqual(Array.from({length:20},()=>a()),Array.from({length:20},()=>b()));
const noMemory=createRuntime(bundle.models.find(m=>m.id==='no_memory-11'));
noMemory.step(1);const first=Array.from(noMemory.step(4));noMemory.reset();noMemory.step(3);
assert.deepEqual(first,Array.from(noMemory.step(4)));
const filtered=samplingProbabilities([1,3,2],{temperature:.8,topK:1});
assert.deepEqual(Array.from(filtered),[0,1,0]);
// The UI can carry state beyond the 32-character evaluation windows. Exercise
// its maximum prompt + generated length on every exported checkpoint.
for(const record of bundle.models) {
  const runtime=createRuntime(record),random=seededRandom(123);
  const prefix='First Citizen:\n'.repeat(35).slice(0,512);
  let logits;
  for(const token of encodePrompt(prefix,bundle.vocabulary))logits=runtime.step(token);
  for(let position=0;position<800;position++) {
    assert.ok(Array.from(logits).every(Number.isFinite),`Non-finite logits: ${record.id} at ${position}`);
    const token=sampleToken(logits,{temperature:.8,topK:20},random);
    assert.ok(token>=0&&token<bundle.vocabulary.length);
    logits=runtime.step(token);
  }
  assert.ok(flat(runtime.state()).every(Number.isFinite),`Non-finite state: ${record.id}`);
}
console.log(JSON.stringify({checkpoint_models:bundle.models.length,parity_cases:fixtures.length,maxLogitError,maxStateError,long_generation_cases:bundle.models.length,passed:true},null,2));
