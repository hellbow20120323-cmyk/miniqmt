(function () {
  const API = "";

  function get(path) {
    return fetch(API + path).then((r) => {
      if (!r.ok) throw new Error(r.status + " " + r.statusText);
      return r.json();
    });
  }

  // --- Tabs ---
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", function () {
      const tab = this.dataset.tab;
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      this.classList.add("active");
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      document.getElementById("panel-" + tab).classList.add("active");
      if (tab === "overview") {
        stopOverviewAutoRefresh();
        loadOverview();
        startOverviewAutoRefresh();
      }
      if (tab === "history") {
        stopOverviewAutoRefresh();
        loadKline(document.querySelector(".kline-tab.active").dataset.symbol);
        loadSignals();
      }
    });
  });

  // --- Overview ---
  function formatNum(n) {
    if (n == null || n === "") return "—";
    const x = Number(n);
    if (Number.isNaN(x)) return "—";
    if (Number.isInteger(x)) return x.toLocaleString();
    return x.toFixed(3);
  }

  function formatPct(n) {
    if (n == null || n === "") return "—";
    const x = Number(n);
    if (Number.isNaN(x)) return "—";
    const s = x >= 0 ? "+" + x.toFixed(2) + "%" : x.toFixed(2) + "%";
    return s;
  }

  function renderSymbolCard(symbolKey, data) {
    const el = document.getElementById("card-" + symbolKey);
    if (!el || !data) {
      if (el) el.innerHTML = "<p class=\"text-muted\">无数据</p>";
      return;
    }
    const q = data.quote || {};
    const st = data.state || {};
    const atr = data.atr_info || {};
    const price = q.price ?? atr.curr_p ?? 0;
    const preClose = q.preClose;
    const pct = preClose && preClose > 0 && price > 0 ? (price / preClose - 1) * 100 : null;
    const pos = q.position || {};
    const vol = pos.volume || 0;
    const usable = pos.can_use_volume || 0;
    const openP = pos.open_price;
    const mv = pos.market_value;
    const cost = vol * (openP || 0);
    const pnl = mv != null && cost > 0 ? mv - cost : null;
    const pnlPct = cost > 0 && pnl != null ? (pnl / cost) * 100 : null;
    const fixedVol = st.fixed_volume || 0;
    const flowEst = Math.max(vol - fixedVol, 0);

    const mtime = data.quote_mtime;
    const stale = data.quote_stale || (atr && atr.data_stale);
    let updateTimeStr = q.time || "";
    if (!updateTimeStr && mtime) {
      try {
        updateTimeStr = new Date(mtime * 1000).toLocaleTimeString("zh-CN", { hour12: false });
      } catch (e) {
        // ignore
      }
    }

    let html = "<h3>" + (data.name || symbolKey) + "</h3>";
    html += "<div class=\"grid-2\">";
    html += "<span class=\"label\">当前价格</span><span class=\"value\">" + formatNum(price) + "</span>";
    html += "<span class=\"label\">当日涨跌幅</span><span class=\"value " + (pct != null && pct >= 0 ? "up" : "down") + "\">" + formatPct(pct) + "</span>";
    html += "<span class=\"label\">最新更新时间</span><span class=\"value\">" + (updateTimeStr || "—") + "</span>";
    if (stale) {
      html += "<span class=\"label\">信号状态</span><span class=\"value down\">暂停触发（行情过期）</span>";
    } else {
      html += "<span class=\"label\">信号状态</span><span class=\"value up\">正常</span>";
    }
    html += "<span class=\"label\">持仓(桥)</span><span class=\"value\">" + vol + " 股 (可用 " + usable + ")</span>";
    if (vol > 0 && openP != null) {
      html += "<span class=\"label\">持仓成本</span><span class=\"value\">" + formatNum(openP) + "</span>";
      if (mv != null) {
        html += "<span class=\"label\">持仓市值</span><span class=\"value\">" + formatNum(mv) + "</span>";
        html += "<span class=\"label\">浮动盈亏</span><span class=\"value " + (pnl >= 0 ? "up" : "down") + "\">" + formatNum(pnl) + (pnlPct != null ? " (" + formatPct(pnlPct) + ")" : "") + "</span>";
      }
    }
    if (fixedVol > 0) {
      html += "<span class=\"label\">固定/流动仓(估)</span><span class=\"value\">" + fixedVol + " / " + flowEst + " 股</span>";
    }
    html += "<span class=\"label\">交易基准价</span><span class=\"value\">" + formatNum(data.last_buy_price) + "</span>";
    if (data.next_buy_price != null) {
      html += "<span class=\"label\">下跌买</span><span class=\"value\">" + formatNum(data.next_buy_price) + " / " + data.next_buy_shares + " 股</span>";
    }
    if (data.next_sell_price != null) {
      html += "<span class=\"label\">上涨卖</span><span class=\"value\">" + formatNum(data.next_sell_price) + " / " + data.next_sell_shares + " 股</span>";
    }
    const kCount = (q.history && q.history.length) || 0;
    html += "<span class=\"label\">K 线数量</span><span class=\"value\">" + kCount + " 根</span>";
    if (atr.atr14 != null) {
      html += "<span class=\"label\">ATR(14)</span><span class=\"value\">" + formatNum(atr.atr14) + "</span>";
      html += "<span class=\"label\">趋势</span><span class=\"value\">" + (atr.trend || "—") + "</span>";
      html += "<span class=\"label\">网格步长</span><span class=\"value\">" + (atr.grid_step != null ? (atr.grid_step * 100).toFixed(2) + "%" : "—") + "</span>";
    }
    html += "<span class=\"label\">T+0 层数/持仓</span><span class=\"value\">" + (data.hold_layers || 0) + " 层 / " + (data.hold_t0_volume || 0) + " 股</span>";
    if (atr.pool_committed != null) {
      const pctPool = (atr.pool_committed / 300000) * 100;
      html += "<span class=\"label\">共享池占用</span><span class=\"value\">" + formatNum(Math.round(atr.pool_committed)) + " (" + pctPool.toFixed(1) + "%)</span>";
    }
    html += "</div>";
    el.innerHTML = html;
  }

  function renderPool(pool) {
    const el = document.getElementById("pool-content");
    if (!el) return;
    const u159 = pool.used_159201 ?? 0;
    const f159 = pool.frozen_159201 ?? 0;
    const u512 = pool.used_512890 ?? 0;
    const f512 = pool.frozen_512890 ?? 0;
    const committed = pool.committed ?? u159 + f159 + u512 + f512;
    let html = "<div class=\"grid-2\">";
    html += "<span class=\"label\">159201 已用/冻结</span><span class=\"value\">" + formatNum(u159) + " / " + formatNum(f159) + "</span>";
    if (u512 !== undefined || f512 !== undefined) {
      html += "<span class=\"label\">512890 已用/冻结</span><span class=\"value\">" + formatNum(u512) + " / " + formatNum(f512) + "</span>";
    }
    html += "<span class=\"label\">合计占用</span><span class=\"value\">" + formatNum(committed) + "</span>";
    html += "</div>";
    el.innerHTML = html;
  }

  function renderSignal(signal, result) {
    const el = document.getElementById("signal-content");
    if (!el) return;
    if (!signal || !signal.signal_id) {
      el.innerHTML = "<p class=\"text-muted\">暂无信号</p>";
      return;
    }
    let html = "<pre>";
    html += "信号: " + (signal.direction || "") + " " + (signal.code || "") + " " + (signal.shares || 0) + " 股 @ " + (signal.price || "") + "\n";
    if (result && result.signal_id) {
      html += "结果: " + (result.status || "") + (result.message ? " — " + result.message : "") + "\n";
    }
    html += "</pre>";
    el.innerHTML = html;
  }

  const OVERVIEW_REFRESH_SEC = 10;
  let overviewRefreshTimer = null;

  function loadOverview(silent) {
    const loading = document.getElementById("overview-loading");
    const content = document.getElementById("overview-content");
    const errEl = document.getElementById("overview-error");
    if (!silent) {
      loading.hidden = false;
      content.hidden = true;
      errEl.hidden = true;
    }
    get("/api/overview")
      .then((res) => {
        loading.hidden = true;
        content.hidden = false;
        renderSymbolCard("159201", res.symbols && res.symbols["159201"]);
        renderSymbolCard("512890", res.symbols && res.symbols["512890"]);
        renderPool(res.pool || {});
        renderSignal(res.order_signal || {}, res.order_result || {});
        if (silent) errEl.hidden = true;
      })
      .catch((e) => {
        loading.hidden = true;
        if (!silent) {
          errEl.textContent = "加载失败: " + e.message;
          errEl.hidden = false;
        }
      });
  }

  function startOverviewAutoRefresh() {
    if (overviewRefreshTimer) return;
    overviewRefreshTimer = setInterval(function () {
      if (document.getElementById("panel-overview").classList.contains("active")) {
        loadOverview(true);
      }
    }, OVERVIEW_REFRESH_SEC * 1000);
  }

  function stopOverviewAutoRefresh() {
    if (overviewRefreshTimer) {
      clearInterval(overviewRefreshTimer);
      overviewRefreshTimer = null;
    }
  }

  // --- History K-line ---
  let klineChart = null;

  document.querySelectorAll(".kline-tab").forEach((btn) => {
    btn.addEventListener("click", function () {
      document.querySelectorAll(".kline-tab").forEach((b) => b.classList.remove("active"));
      this.classList.add("active");
      loadKline(this.dataset.symbol);
    });
  });

  function loadKline(symbol) {
    const container = document.getElementById("kline-container");
    if (!container) return;
    get("/api/history/kline/" + symbol)
      .then((res) => {
        const kline = res.kline || [];
        const times = kline.map((r) => r.time);
        const ohlc = kline.map((r) => [r.open, r.close, r.low, r.high]);
        if (!klineChart) {
          klineChart = echarts.init(container);
        }
        klineChart.setOption({
          backgroundColor: "transparent",
          tooltip: { trigger: "axis" },
          xAxis: { type: "category", data: times, axisLabel: { fontSize: 10 } },
          yAxis: { type: "value", scale: true, splitLine: { lineStyle: { opacity: 0.2 } } },
          series: [
            { type: "candlestick", data: ohlc, itemStyle: { color: "#3fb950", color0: "#f85149", borderColor: "#3fb950", borderColor0: "#f85149" } }
          ],
          grid: { left: "10%", right: "8%", top: "10%", bottom: "15%" }
        }, true);
      })
      .catch((e) => {
        if (klineChart) klineChart.setOption({ title: { text: "加载失败: " + e.message, left: "center" } }, true);
      });
  }

  // --- History signals ---
  function loadSignals() {
    const filter = document.getElementById("signal-symbol-filter");
    const symbol = filter ? filter.value : "";
    const loading = document.getElementById("signals-loading");
    const table = document.getElementById("signals-table");
    const tbody = table ? table.querySelector("tbody") : null;
    const errEl = document.getElementById("signals-error");
    if (loading) loading.hidden = false;
    if (table) table.hidden = true;
    if (errEl) errEl.hidden = true;
    const url = "/api/history/signals" + (symbol ? "?symbol=" + encodeURIComponent(symbol) : "");
    get(url)
      .then((res) => {
        if (loading) loading.hidden = true;
        const signals = res.signals || [];
        if (tbody) {
          tbody.innerHTML = "";
          signals.forEach((s) => {
            const tr = document.createElement("tr");
            const timeStr = s._time_str || (s.timestamp != null ? new Date(s.timestamp * 1000).toLocaleString() : "—");
            const dir = (s.direction || "").toUpperCase();
            tr.innerHTML =
              "<td>" + timeStr + "</td>" +
              "<td>" + (s.code || "—") + "</td>" +
              "<td class=\"dir-" + (dir === "BUY" ? "buy" : "sell") + "\">" + dir + "</td>" +
              "<td>" + formatNum(s.price) + "</td>" +
              "<td>" + formatNum(s.shares) + "</td>" +
              "<td>" + (s.reason || "—") + "</td>";
            tbody.appendChild(tr);
          });
        }
        if (table) table.hidden = false;
      })
      .catch((e) => {
        if (loading) loading.hidden = true;
        if (errEl) {
          errEl.textContent = "加载失败: " + e.message;
          errEl.hidden = false;
        }
      });
  }

  document.getElementById("signal-symbol-filter")?.addEventListener("change", loadSignals);

  // --- Init ---
  loadOverview();
  startOverviewAutoRefresh();
})();
