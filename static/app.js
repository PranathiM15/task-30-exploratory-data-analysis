const $ = (id) => document.getElementById(id);
let activeFile = "";
let lastAnalysis = null;
const formatter = new Intl.NumberFormat("en-IN");
const number = (value) => value === null || value === undefined ? "—" : formatter.format(value);
const decimal = (value) => value === null || value === undefined ? "—" : Number(value).toLocaleString("en-IN", {maximumFractionDigits: 2});

function setStatus(message, tone="") {
  const line = $("statusLine"); line.textContent = message; line.className = `status-line ${tone}`;
}

async function getJSON(url, options={}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Something went wrong.");
  return data;
}

async function init() {
  try {
    const result = await getJSON("/api/list");
    const select = $("datasetSelect");
    select.innerHTML = "";
    result.files.forEach((item) => {
      const option = document.createElement("option"); option.value = item.name;
      option.textContent = item.name.replaceAll("_", " ").replace(".csv", "");
      select.appendChild(option);
    });
    if (!result.files.length) throw new Error("No CSV files found in the datasets folder.");
    select.addEventListener("change", () => loadDataset(select.value));
    await loadDataset(result.files[0].name);
  } catch (error) { setStatus(error.message, "error"); }
}

async function loadDataset(filename) {
  activeFile = filename; setStatus(`Analyzing ${filename}…`);
  try {
    const data = await getJSON(`/api/analyze?file=${encodeURIComponent(filename)}`);
    lastAnalysis = data; render(data);
    const sampleNote = filename.includes("synthetic") ? " · SYNTHETIC TEACHING SAMPLE" : "";
    setStatus(`Analysis complete · ${number(data.rows)} rows · ${data.columns} columns · ${data.memory_mb} MB in memory${sampleNote}`, "good");
  } catch (error) { setStatus(error.message, "error"); }
}

function render(data) {
  $("rowsValue").textContent = number(data.rows);
  $("colsValue").textContent = number(data.columns);
  $("missingValue").textContent = number(data.missing_total);
  $("duplicateValue").textContent = number(data.duplicate_rows);
  $("missingFoot").textContent = data.missing_total ? "Needs a cleaning review" : "No missing cells detected";
  $("schemaCaption").textContent = `${data.column_info.length} FIELDS · ${data.numeric_count} NUMERIC · ${data.categorical_count} OTHER`;
  $("observations").innerHTML = "";
  data.observations.forEach((note) => { const li = document.createElement("li"); li.textContent = note; $("observations").appendChild(li); });
  const chartMap = {hist:"chartHist",missing:"chartMissing",category:"chartCategory",corr:"chartCorr"};
  Object.entries(chartMap).forEach(([key,id]) => { $(id).src = `${data.charts[key]}?v=${Date.now()}`; });
  renderSchema(data.column_info); renderStats(data.numeric_stats); renderPreview(data.preview);
  $("cleanResult").textContent = "";
}

function renderSchema(rows) {
  const body = $("schemaBody"); body.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const percent = Math.max(0, 100 - row.missing_pct);
    tr.innerHTML = `<td><b>${escapeHTML(row.name)}</b></td><td><span class="type-tag">${escapeHTML(row.dtype)}</span></td><td>${number(row.missing)}</td><td><div class="completeness"><div class="bar-track"><div class="bar-fill ${percent < 90 ? 'bad-fill' : ''}" style="width:${percent}%"></div></div><span>${percent.toFixed(1)}%</span></div></td><td>${number(row.unique)}</td>`;
    body.appendChild(tr);
  });
}

function renderStats(rows) {
  const body = $("statsBody"); body.innerHTML = "";
  if (!rows.length) { body.innerHTML = '<tr><td colspan="7" class="empty">No numeric columns available in this dataset.</td></tr>'; return; }
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    [row.column, number(row.count), decimal(row.mean), decimal(row.std), decimal(row.min), decimal(row.median), decimal(row.max)].forEach((value) => {
      const td = document.createElement("td"); td.textContent = value; tr.appendChild(td);
    }); body.appendChild(tr);
  });
}

function renderPreview(rows) {
  const head = $("previewHead"), body = $("previewBody"); head.innerHTML=""; body.innerHTML="";
  if (!rows.length) { body.innerHTML='<tr><td class="empty">No preview rows available.</td></tr>'; return; }
  const columns = Object.keys(rows[0]);
  const hr = document.createElement("tr"); columns.forEach((col) => { const th=document.createElement("th"); th.textContent=col; hr.appendChild(th); }); head.appendChild(hr);
  rows.forEach((row) => { const tr=document.createElement("tr"); columns.forEach((col) => { const td=document.createElement("td"); td.textContent=row[col] === null ? "—" : String(row[col]); tr.appendChild(td); }); body.appendChild(tr); });
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));
}

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) { setStatus("Please choose a .csv file.", "error"); return; }
  setStatus(`Uploading ${file.name}…`);
  try {
    const uploaded = await getJSON(`/api/upload?name=${encodeURIComponent(file.name)}`, {method:"POST",headers:{"Content-Type":"text/csv"},body:file});
    const result = await getJSON("/api/list"); const select=$("datasetSelect");
    select.innerHTML=""; result.files.forEach((item)=>{const option=document.createElement("option");option.value=item.name;option.textContent=item.name.replaceAll("_"," ").replace(".csv","");select.appendChild(option);});
    select.value=uploaded.file; await loadDataset(uploaded.file); setStatus(`Uploaded and analyzed ${uploaded.file}`, "good");
  } catch(error) { setStatus(error.message, "error"); }
}

async function cleanCurrent() {
  if (!activeFile) return;
  const buttons=[$("cleanBtn"),$("cleanBtnBottom")]; buttons.forEach(b=>{b.disabled=true;b.style.opacity=".65";});
  setStatus("Cleaning the selected dataset…");
  try {
    const result=await getJSON("/api/clean",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({file:activeFile})});
    const link=document.createElement("a"); link.href=`/api/download-clean?file=${encodeURIComponent(result.file)}`; link.download=result.file; document.body.appendChild(link); link.click(); link.remove();
    const msg=`Saved ${result.file} · ${number(result.duplicates_removed)} duplicates removed · ${number(result.missing_filled)} missing cells filled`;
    $("cleanResult").textContent=msg; setStatus(msg,"good");
  } catch(error) { setStatus(error.message,"error"); }
  finally { buttons.forEach(b=>{b.disabled=false;b.style.opacity="1";}); }
}

$("csvUpload").addEventListener("change", (event) => uploadFile(event.target.files[0]));
$("cleanBtn").addEventListener("click", cleanCurrent);
$("cleanBtnBottom").addEventListener("click", cleanCurrent);
init();
