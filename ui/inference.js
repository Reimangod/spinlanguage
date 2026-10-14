/** Actual checkpoint inference: reduced YIG forward and PyTorch GRU equations.
 * Float64 browser arithmetic approximates the original float32/complex64 run.
 * See tests/browser_inference.mjs for checkpoint-logit and state parity.
 */
export function softmax(logits, temperature=1) {
  const maximum=Math.max(...logits);
  const values=Float64Array.from(logits,v=>Math.exp((v-maximum)/temperature));
  const total=values.reduce((a,b)=>a+b,0);
  return values.map(v=>v/total);
}

export function seededRandom(seed) {
  let state=seed>>>0;
  return ()=>{state=(state+0x6D2B79F5)>>>0;let t=state;t=Math.imul(t^(t>>>15),t|1);
    t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
}

export function samplingProbabilities(logits,{temperature=.8,topK=0,greedy=false}={}) {
  if(greedy){const p=new Float64Array(logits.length);p[logits.indexOf(Math.max(...logits))]=1;return p;}
  const values=Array.from(logits);
  if(topK>0 && topK<values.length) {
    const order=values.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v).slice(0,topK);
    const keep=new Set(order.map(x=>x.i));
    for(let i=0;i<values.length;i++)if(!keep.has(i))values[i]=-Infinity;
  }
  return softmax(values,temperature);
}

export function sampleToken(logits,options={},random=Math.random) {
  const probabilities=samplingProbabilities(logits,options);
  let draw=random();
  for(let i=0;i<probabilities.length;i++){draw-=probabilities[i];if(draw<0)return i;}
  return probabilities.length-1;
}

function readout(features,weights) {
  const matrix=weights['readout.weight'],bias=weights['readout.bias'];
  return Float64Array.from(matrix,(row,i)=>row.reduce((sum,w,j)=>sum+w*features[j],bias[i]));
}

export class PhysicalRuntime {
  constructor(record) {
    this.record=record;this.weights=record.weights;this.config=record.physics;
    const c=this.config,w=this.weights;
    if(c.thermal_noise || c.sensor_noise_std || c.dac_bits)throw new Error('UI supports the original deterministic checkpoints only.');
    this.dt=c.token_duration_s/c.substeps;
    this.inputDecay=Math.exp(-2*Math.PI*c.input_bandwidth_Hz*this.dt/2);
    this.sensorDecay=Math.exp(-2*Math.PI*c.sensor_bandwidth_Hz*this.dt);
    this.nu=c.nonlinear_scale*c.gyromagnetic_rad_s_T*(4*Math.PI*1e-7)*c.saturation_magnetization_A_m/2;
    this.currents=w['reservoir.raw_current'].map(v=>c.current_limit_A*Math.tanh(v));
    this.field=w['reservoir.coil_field_T_A'].map(row=>row.reduce((s,v,i)=>s+v*this.currents[i],0));
    // ifft(diagonal propagator) gives the exact periodic convolution kernel.
    const propagator=w['reservoir.propagator_half'];
    this.kernel=new Float64Array(c.cells*2);
    for(let j=0;j<c.cells;j++)for(let k=0;k<c.cells;k++) {
      const angle=2*Math.PI*j*k/c.cells,cos=Math.cos(angle),sin=Math.sin(angle);
      this.kernel[2*j]+=(propagator[k][0]*cos-propagator[k][1]*sin)/c.cells;
      this.kernel[2*j+1]+=(propagator[k][0]*sin+propagator[k][1]*cos)/c.cells;
    }
    this.reset();
  }
  reset() {
    this.amplitude=new Float64Array(this.config.cells*2);
    this.rf=new Float64Array(this.config.input_ports*2);
    this.sensor=new Float64Array(this.config.sensor_ports*2);
  }
  propagate(input) {
    const n=this.config.cells,output=new Float64Array(n*2),kernel=this.kernel;
    for(let i=0;i<n;i++)for(let j=0;j<n;j++) {
      const k=2*((i-j+n)%n),p=2*j;
      output[2*i]+=kernel[k]*input[p]-kernel[k+1]*input[p+1];
      output[2*i+1]+=kernel[k]*input[p+1]+kernel[k+1]*input[p];
    }
    return output;
  }
  step(token) {
    if(this.record.variant==='no_memory')this.reset();
    const c=this.config,w=this.weights,desired=w['reservoir.token_code'][token];
    if(!desired)throw new Error('Invalid character token.');
    const features=[];
    for(let substep=0;substep<c.substeps;substep++) {
      for(let port=0;port<c.input_ports;port++)for(let component=0;component<2;component++) {
        const p=port*2+component;
        this.rf[p]=this.inputDecay*this.rf[p]+(1-this.inputDecay)*desired[port][component];
      }
      const drive=new Float64Array(c.cells*2),profiles=w['reservoir.input_profiles'];
      const scale=c.gyromagnetic_rad_s_T*c.input_field_limit_T/2;
      for(let cell=0;cell<c.cells;cell++)for(let port=0;port<c.input_ports;port++) {
        drive[2*cell]+=this.rf[2*port]*profiles[port][cell]*scale;
        drive[2*cell+1]+=this.rf[2*port+1]*profiles[port][cell]*scale;
      }
      const before=this.amplitude;
      const a=this.propagate(before);
      for(let cell=0;cell<c.cells;cell++) {
        const p=2*cell;
        const real=a[p]+this.dt*drive[p]/2,imag=a[p+1]+this.dt*drive[p+1]/2;
        const phase=this.dt*(c.gyromagnetic_rad_s_T*this.field[cell]+this.nu*(real*real+imag*imag));
        const cos=Math.cos(phase),sin=Math.sin(phase);
        a[p]=real*cos+imag*sin+this.dt*drive[p]/2;
        a[p+1]=imag*cos-real*sin+this.dt*drive[p+1]/2;
      }
      this.amplitude=this.propagate(a);
      for(let port=0;port<c.input_ports;port++)for(let component=0;component<2;component++) {
        const p=port*2+component;
        this.rf[p]=this.inputDecay*this.rf[p]+(1-this.inputDecay)*desired[port][component];
      }
      for(let sensor=0;sensor<c.sensor_ports;sensor++)for(let component=0;component<2;component++) {
        let value=0;
        for(let cell=0;cell<c.cells;cell++) {
          const p=2*cell+component;
          value+=(before[p]+this.amplitude[p])/2*w['reservoir.sensor_profiles'][sensor][cell];
        }
        const p=2*sensor+component;
        this.sensor[p]=this.sensorDecay*this.sensor[p]+(1-this.sensorDecay)*value;
      }
      if((substep+1)%(c.substeps/c.samples_per_token)===0)
        for(const v of this.sensor)features.push(v*c.sensor_scale);
    }
    return readout(features,w);
  }
  state() {
    const pairs=array=>Array.from({length:array.length/2},(_,i)=>[array[2*i],array[2*i+1]]);
    return [pairs(this.amplitude),pairs(this.rf),pairs(this.sensor)];
  }
  visualization() {
    return {amplitude:Array.from({length:this.config.cells},(_,i)=>Math.hypot(this.amplitude[2*i],this.amplitude[2*i+1])),
      currents_mA:this.currents.map(v=>v*1000)};
  }
}

export class GRURuntime {
  constructor(record) {this.record=record;this.weights=record.weights;this.reset();}
  reset() {this.hidden=new Float64Array(this.weights['gru.weight_hh_l0'][0].length);}
  step(token) {
    const w=this.weights,n=this.hidden.length,old=this.hidden;
    const input=w['gru.weight_ih_l0'],recurrent=w['gru.weight_hh_l0'];
    const bi=w['gru.bias_ih_l0'],bh=w['gru.bias_hh_l0'];
    const inValues=new Float64Array(n*3),hiddenValues=new Float64Array(n*3);
    for(let row=0;row<n*3;row++) {
      inValues[row]=input[row][token]+bi[row];
      hiddenValues[row]=recurrent[row].reduce((s,v,i)=>s+v*old[i],bh[row]);
    }
    const sigmoid=x=>1/(1+Math.exp(-x));
    for(let i=0;i<n;i++) {
      const r=sigmoid(inValues[i]+hiddenValues[i]);
      const z=sigmoid(inValues[n+i]+hiddenValues[n+i]);
      const candidate=Math.tanh(inValues[2*n+i]+r*hiddenValues[2*n+i]);
      this.hidden[i]=(1-z)*candidate+z*old[i];
    }
    return readout(this.hidden,w);
  }
  state(){return Array.from(this.hidden);}
  visualization(){return null;}
}

export function createRuntime(record) {return record.variant==='gru'?new GRURuntime(record):new PhysicalRuntime(record);}

export function encodePrompt(prompt,vocabulary) {
  if(!prompt.length)throw new Error('続きを生成する英字の文章を入力してください。');
  const mapping=new Map(vocabulary.map((c,i)=>[c,i]));
  const missing=[...new Set([...prompt].filter(c=>!mapping.has(c)))];
  if(missing.length)throw new Error(`学習語彙にない文字: ${missing.map(c=>JSON.stringify(c)).join(' ')}。英字・対応する記号を使ってください。`);
  return Array.from(prompt,c=>mapping.get(c));
}
