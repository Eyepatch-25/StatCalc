from flask import Flask, render_template, request, jsonify
from scipy.stats import pearsonr, spearmanr
from statistics import median, multimode

app = Flask(__name__, template_folder='pages')


def quartiles(data):
    data = sorted(data)
    n = len(data)
    mid = n // 2
    lower = data[:mid]
    upper = data[mid:] if n % 2 == 0 else data[mid+1:]
    return median(lower), median(upper)


def compute(x, y, predict_x=None, predict_y=None):
    n = len(x)
    x_sqr = [a**2 for a in x]
    y_sqr = [b**2 for b in y]
    xy    = [a*b for a, b in zip(x, y)]

    sumx   = sum(x)
    sumy   = sum(y)
    sum_x2 = sum(x_sqr)
    sum_y2 = sum(y_sqr)
    sum_xy = sum(xy)

    cf_x, cf_y, sx, sy = [], [], 0, 0
    for i in range(n):
        sx += x[i]; sy += y[i]
        cf_x.append(sx); cf_y.append(sy)

    numerator   = n * sum_xy - sumx * sumy
    denominator = ((n * sum_x2 - sumx**2) * (n * sum_y2 - sumy**2))**0.5
    # r_manual    = numerator / denominator if denominator != 0 else 0

    r_pearson,  p_pearson  = pearsonr(x, y)
    r_spearman, p_spearman = spearmanr(x, y)

    rank_x = [sorted(x, reverse=True).index(xi)+1 for xi in x]
    rank_y = [sorted(y, reverse=True).index(yi)+1 for yi in y]
    d   = [rx-ry for rx, ry in zip(rank_x, rank_y)]
    d2  = [di**2 for di in d]
    sum_d2 = sum(d2)
    r_s = 1 - (6*sum_d2)/(n*(n**2-1))

    mean_x = sumx/n
    mean_y = sumy/n
    med_x  = median(x)
    med_y  = median(y)
    mode_x = multimode(x)
    mode_y = multimode(y)
    Q1_x, Q3_x = quartiles(x)
    Q1_y, Q3_y = quartiles(y)

    b_yx = numerator/(n*sum_x2 - sumx**2) if (n*sum_x2 - sumx**2) != 0 else 0
    a_yx = mean_y - b_yx*mean_x
    b_xy = numerator/(n*sum_y2 - sumy**2) if (n*sum_y2 - sumy**2) != 0 else 0
    a_xy = mean_x - b_xy*mean_y

    pred_y = round(a_yx + b_yx*predict_x, 4) if predict_x is not None else None
    pred_x = round(a_xy + b_xy*predict_y, 4) if predict_y is not None else None

    # regression line points for chart
    all_x = list(x) + ([predict_x] if predict_x is not None else [])
    x_min = min(all_x) - 5
    x_max = max(all_x) + 5
    line_pts = [{"x": round(xv,2), "y": round(a_yx + b_yx*xv, 2)}
                for xv in [x_min + i*(x_max-x_min)/99 for i in range(100)]]

    rows = []
    for i in range(n):
        rows.append({
            "i": i+1, "x": x[i], "y": y[i],
            "cf_x": cf_x[i], "cf_y": cf_y[i],
            "x2": x_sqr[i], "y2": y_sqr[i], "xy": xy[i],
            "rx": rank_x[i], "ry": rank_y[i], "d": d[i], "d2": d2[i]
        })

    return {
        "rows": rows,
        "sums": {"x": sumx,"y": sumy,"x2": sum_x2,"y2": sum_y2,"xy": sum_xy,"d2": sum_d2},
        "stats": {
            "mean_x": round(mean_x,4), "mean_y": round(mean_y,4),
            "median_x": med_x, "median_y": med_y,
            "mode_x": mode_x, "mode_y": mode_y,
            "Q1_x": Q1_x, "Q3_x": Q3_x,
            "Q1_y": Q1_y, "Q3_y": Q3_y,
        },
        "correlation": {
            # "r_manual":   round(r_manual,4),
            "r_pearson":  round(r_pearson,4),  "p_pearson":  round(p_pearson,4),
            "r_spearman": round(r_spearman,4), "p_spearman": round(p_spearman,4),
            # "r_s_manual": round(r_s,4), "sum_d2": sum_d2,
        },
        "regression": {
            "a_yx": round(a_yx,4), "b_yx": round(b_yx,4),
            "a_xy": round(a_xy,4), "b_xy": round(b_xy,4),
        },
        "prediction": {
            "predict_x": predict_x, "pred_y": pred_y,
            "predict_y": predict_y, "pred_x": pred_x,
        },
        "chart": {
            "scatter": [{"x": x[i], "y": y[i]} for i in range(n)],
            "line": line_pts,
            "pred_point": {"x": predict_x, "y": pred_y} if predict_x is not None else None,
        }
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compute", methods=["POST"])
def run_compute():
    data = request.get_json()
    try:
        x = [float(v) for v in data["x"]]
        y = [float(v) for v in data["y"]]
        if len(x) != len(y):
            return jsonify({"error": "x and y must have the same length"}), 400
        px = float(data["predict_x"]) if data.get("predict_x") not in (None, "") else None
        py = float(data["predict_y"]) if data.get("predict_y") not in (None, "") else None
        return jsonify(compute(x, y, px, py))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5050)