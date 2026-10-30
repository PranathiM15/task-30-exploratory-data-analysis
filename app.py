from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json, mimetypes, re, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE / "datasets"
CLEAN = BASE / "cleaned"
CHARTS = BASE / "charts"
STATIC = BASE / "static"
for folder in (DATA, CLEAN, CHARTS, STATIC):
    folder.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#111821", "axes.facecolor": "#111821",
    "axes.edgecolor": "#314152", "axes.labelcolor": "#c7d2df",
    "xtick.color": "#96a6b8", "ytick.color": "#96a6b8",
    "text.color": "#e8eef5", "grid.color": "#253343",
    "font.family": "DejaVu Sans"
})
ACCENT = "#9fe870"
CYAN = "#59d5d2"
GOLD = "#f4bd67"


def safe_name(name):
    name = Path(name).name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,90}\.csv", name, flags=re.I):
        raise ValueError("Use a CSV filename with letters, numbers, spaces, dot, underscore or dash.")
    return name


def load_csv(path):
    try:
        frame = pd.read_csv(path, low_memory=False)
        if len(frame.columns) == 1:
            frame = pd.read_csv(path, sep=None, engine="python", low_memory=False)
        return frame
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def clean_frame(df):
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", " ") for c in out.columns]
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    before = len(out)
    out = out.drop_duplicates().copy()
    duplicates_removed = before - len(out)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if out[col].isna().any():
                median = out[col].median()
                out[col] = out[col].fillna(0 if pd.isna(median) else median)
        elif out[col].isna().any():
            mode = out[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            out[col] = out[col].fillna(fill)
    return out, duplicates_removed


def make_chart(df, file_slug, chart_type):
    path = CHARTS / f"{file_slug}_{chart_type}.png"
    fig, ax = plt.subplots(figsize=(7.3, 3.1), dpi=120)
    nums = list(df.select_dtypes(include=[np.number]).columns)
    cats = [c for c in df.columns if (df[c].dtype == "object" or str(df[c].dtype) == "bool") and 1 < df[c].nunique(dropna=True) <= 30]
    if chart_type == "hist":
        if nums and df[nums[0]].notna().any():
            col = nums[0]
            ax.hist(df[col].dropna(), bins=18, color=ACCENT, edgecolor="#111821", alpha=.92)
            ax.set_title(f"Distribution · {col}", loc="left", fontsize=12, pad=12, weight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Records")
            ax.grid(axis="y", alpha=.25)
        else:
            ax.text(.5,.5,"No numeric columns available",ha="center",va="center",color="#a9b7c7")
            ax.set_axis_off()
    elif chart_type == "missing":
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].head(12)
        if len(miss):
            ax.barh(miss.index[::-1], miss.values[::-1], color=GOLD)
            ax.set_title("Missing values by column", loc="left", fontsize=12, pad=12, weight="bold")
            ax.set_xlabel("Missing cells")
            ax.grid(axis="x", alpha=.25)
        else:
            ax.set_axis_off()
            ax.text(.5,.58,"✓  No missing values detected",ha="center",va="center",fontsize=14,color=ACCENT,weight="bold")
            ax.text(.5,.39,"Every column is complete in this dataset",ha="center",va="center",fontsize=9,color="#a9b7c7")
    elif chart_type == "category":
        if cats:
            col = cats[0]
            counts = df[col].fillna("Missing").astype(str).value_counts().head(8)
            ax.barh(counts.index[::-1], counts.values[::-1], color=CYAN)
            ax.set_title(f"Most common values · {col}", loc="left", fontsize=12, pad=12, weight="bold")
            ax.set_xlabel("Records")
            ax.grid(axis="x", alpha=.25)
        elif nums:
            col = nums[0]
            counts = df[col].round(0).value_counts().head(8).sort_values()
            ax.barh([str(x) for x in counts.index], counts.values, color=CYAN)
            ax.set_title(f"Value counts · {col}", loc="left", fontsize=12, pad=12, weight="bold")
        else:
            ax.text(.5,.5,"No suitable category column",ha="center",va="center")
            ax.set_axis_off()
    elif chart_type == "corr":
        if len(nums) >= 2:
            corr = df[nums[:9]].corr(numeric_only=True)
            im = ax.imshow(corr, cmap="YlGnBu", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right", fontsize=7)
            ax.set_yticks(range(len(corr.index)), corr.index, fontsize=7)
            ax.set_title("Numeric correlation map", loc="left", fontsize=12, pad=12, weight="bold")
            fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
        elif nums:
            ax.boxplot(df[nums[0]].dropna(), vert=False, patch_artist=True,
                       boxprops={"facecolor": CYAN}, medianprops={"color":"#111821"})
            ax.set_title(f"Spread · {nums[0]}", loc="left", fontsize=12, pad=12, weight="bold")
        else:
            ax.text(.5,.5,"No numeric columns available",ha="center",va="center")
            ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_color("#314152")
    fig.tight_layout(pad=1.5)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/charts/{path.name}"


def val(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if hasattr(v, "item"):
        try: return v.item()
        except Exception: pass
    return v if isinstance(v, (str,int,float,bool)) else str(v)


def analyze(filename):
    filename = safe_name(filename)
    source = DATA / filename
    if not source.exists():
        raise FileNotFoundError("Dataset was not found in the datasets folder.")
    df = load_csv(source)
    if df.empty or len(df.columns) == 0:
        raise ValueError("This CSV has no data rows or columns.")
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    categorical = [c for c in df.columns if c not in numeric]
    col_info = []
    for c in df.columns:
        missing = int(df[c].isna().sum())
        col_info.append({"name":str(c),"dtype":str(df[c].dtype),"missing":missing,
                         "missing_pct":round(missing / max(len(df),1) * 100, 1),
                         "unique":int(df[c].nunique(dropna=True))})
    numeric_stats = []
    if numeric:
        desc = df[numeric].describe().T
        for c, row in desc.iterrows():
            numeric_stats.append({"column":str(c),"count":int(row.get("count",0)),
                "mean":round(float(row["mean"]),3) if pd.notna(row.get("mean")) else None,
                "std":round(float(row["std"]),3) if pd.notna(row.get("std")) else None,
                "min":round(float(row["min"]),3) if pd.notna(row.get("min")) else None,
                "q1":round(float(row["25%"]),3) if pd.notna(row.get("25%")) else None,
                "median":round(float(row["50%"]),3) if pd.notna(row.get("50%")) else None,
                "q3":round(float(row["75%"]),3) if pd.notna(row.get("75%")) else None,
                "max":round(float(row["max"]),3) if pd.notna(row.get("max")) else None})
    cat_stats=[]
    for c in categorical[:12]:
        counts = df[c].fillna("Missing").astype(str).value_counts().head(5)
        cat_stats.append({"column":str(c),"top":[{"value":str(k),"count":int(v)} for k,v in counts.items()]})
    duplicate_count = int(df.duplicated().sum())
    missing_total = int(df.isna().sum().sum())
    observations = [f"The dataset contains {len(df):,} rows and {len(df.columns)} columns, giving {len(df)*len(df.columns):,} individual cells."]
    if missing_total:
        topmiss = df.isna().sum().sort_values(ascending=False)
        topcol = str(topmiss.index[0])
        observations.append(f"There are {missing_total:,} missing cells; '{topcol}' has the largest gap ({int(topmiss.iloc[0])}).")
    else:
        observations.append("No missing values were detected, so the original dataset is complete across its columns.")
    observations.append(f"The dataset contains {duplicate_count:,} duplicate rows that should be reviewed before modeling.")
    if categorical:
        cat = categorical[0]
        counts = df[cat].fillna("Missing").astype(str).value_counts()
        if not counts.empty:
            observations.append(f"In '{cat}', the most frequent value is '{counts.index[0]}' ({int(counts.iloc[0]):,} records).")
    if numeric_stats:
        widest = max(numeric_stats, key=lambda x: (x["std"] or 0))
        observations.append(f"'{widest['column']}' has the largest numeric standard deviation ({widest['std']}); its values are comparatively spread out.")
    while len(observations) < 5:
        observations.append("Review the column profile and preview table to identify additional patterns relevant to the dataset context.")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem)[:60]
    charts = {kind:make_chart(df, slug, kind) for kind in ["hist","missing","category","corr"]}
    preview = []
    for _, row in df.head(10).iterrows():
        preview.append({str(k):val(v) for k,v in row.items()})
    return {"file":filename,"rows":int(df.shape[0]),"columns":int(df.shape[1]),
        "numeric_count":len(numeric),"categorical_count":len(categorical),
        "missing_total":missing_total,"duplicate_rows":duplicate_count,
        "memory_mb":round(float(df.memory_usage(deep=True).sum())/1024/1024,3),
        "column_info":col_info,"numeric_stats":numeric_stats,"categorical_stats":cat_stats,
        "observations":observations[:5],"charts":charts,"preview":preview}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE), **kwargs)

    def log_message(self, fmt, *args):
        print("[EDA] " + (fmt % args))

    def send_json(self, payload, status=200):
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Cache-Control","no-store")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        parsed=urlparse(self.path)
        if parsed.path == "/api/list":
            files=[]
            preferred = ["iris.csv", "titanic_sample_synthetic.csv", "student_performance_sample_synthetic.csv"]
            paths = list(DATA.glob("*.csv"))
            paths.sort(key=lambda item: preferred.index(item.name) if item.name in preferred else len(preferred))
            for p in paths:
                files.append({"name":p.name,"size_kb":round(p.stat().st_size/1024,1)})
            return self.send_json({"files":files})
        if parsed.path == "/api/analyze":
            try:
                q=parse_qs(parsed.query); name=q.get("file",[""])[0]
                return self.send_json(analyze(name))
            except Exception as e:
                return self.send_json({"error":str(e)},400)
        if parsed.path == "/api/download-clean":
            q=parse_qs(parsed.query); name=Path(q.get("file",[""])[0]).name
            p=CLEAN/name
            if not p.exists(): return self.send_json({"error":"Cleaned file not found. Run Clean & Export first."},404)
            data=p.read_bytes(); self.send_response(200)
            self.send_header("Content-Type","text/csv; charset=utf-8")
            self.send_header("Content-Disposition",f'attachment; filename="{name}"')
            self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
            return
        if parsed.path == "/":
            self.path="/static/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed=urlparse(self.path)
        if parsed.path == "/api/upload":
            try:
                name=safe_name(parse_qs(parsed.query).get("name",["upload.csv"])[0])
                length=int(self.headers.get("Content-Length","0"))
                if length <= 0 or length > 15*1024*1024:
                    raise ValueError("Choose a CSV file smaller than 15 MB.")
                raw=self.rfile.read(length)
                if b"\x00" in raw[:2048]: raise ValueError("The uploaded file does not look like a text CSV.")
                target=DATA/name
                target.write_bytes(raw)
                # Validate that pandas can read it before reporting success.
                test=load_csv(target)
                if test.empty or len(test.columns)==0:
                    target.unlink(missing_ok=True); raise ValueError("The uploaded CSV has no usable rows or columns.")
                return self.send_json({"ok":True,"file":name})
            except Exception as e:
                return self.send_json({"error":str(e)},400)
        if parsed.path == "/api/clean":
            try:
                length=int(self.headers.get("Content-Length","0")); body=json.loads(self.rfile.read(length) or b"{}")
                name=safe_name(body.get("file","")); source=DATA/name
                if not source.exists(): raise FileNotFoundError("Dataset not found.")
                original=load_csv(source); cleaned, removed=clean_frame(original)
                outname=f"{Path(name).stem}_cleaned.csv"; dest=CLEAN/outname
                cleaned.to_csv(dest,index=False)
                return self.send_json({"ok":True,"file":outname,"rows_before":len(original),"rows_after":len(cleaned),
                    "duplicates_removed":removed,"missing_filled":int(original.isna().sum().sum())})
            except Exception as e:
                return self.send_json({"error":str(e)},400)
        return self.send_json({"error":"Unknown endpoint."},404)


def main():
    host="127.0.0.1"; port=8000
    server=ThreadingHTTPServer((host,port),Handler)
    print("\n  LUMEN / DATA LAB is running")
    print(f"  Open: http://{host}:{port}")
    print("  Stop the server with Ctrl + C\n")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nServer stopped.")
    finally: server.server_close()

if __name__ == "__main__":
    main()
