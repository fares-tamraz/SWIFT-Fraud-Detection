(function () {
  const thresholdEl = document.getElementById("threshold");
  const thresholdValueEl = document.getElementById("thresholdValue");
  const thresholdHintEl = document.getElementById("thresholdHint");
  const thresholdDetailEl = document.getElementById("thresholdDetail");

  const thresholdHints = {
    0.2: "Catch more fraud — more false alarms (flag when probability ≥ 20%).",
    0.3: "Leaning toward catching more fraud; false alarms will increase.",
    0.4: "Slightly sensitive — good for not missing fraud, some false positives.",
    0.5: "Balanced — equal weight to catching fraud and avoiding false alarms.",
    0.6: "Slightly strict — fewer false alarms, may miss some fraud.",
    0.7: "Strict — fewer false alarms; only high-confidence cases flagged.",
    0.8: "Very strict — only flag when the model is very confident.",
    0.9: "Maximum strictness — minimal false alarms; may miss fraud.",
  };

  function getThresholdHint(v) {
    const num = parseFloat(v);
    if (num <= 0.25) return thresholdHints[0.2];
    if (num <= 0.35) return thresholdHints[0.3];
    if (num <= 0.45) return thresholdHints[0.4];
    if (num <= 0.55) return thresholdHints[0.5];
    if (num <= 0.65) return thresholdHints[0.6];
    if (num <= 0.75) return thresholdHints[0.7];
    if (num <= 0.85) return thresholdHints[0.8];
    return thresholdHints[0.9];
  }

  function updateThresholdUI() {
    const v = thresholdEl.value;
    thresholdValueEl.textContent = parseFloat(v).toFixed(2);
    thresholdHintEl.textContent = getThresholdHint(v);
    thresholdDetailEl.textContent =
      "At " + parseFloat(v).toFixed(2) + ": transactions with fraud probability ≥ " + (parseFloat(v) * 100).toFixed(0) + "% are flagged as fraud.";
  }

  if (thresholdEl) {
    thresholdEl.addEventListener("input", updateThresholdUI);
    updateThresholdUI();
  }

  // Single form
  const singleForm = document.getElementById("singleForm");
  const singleError = document.getElementById("singleError");
  const singleResult = document.getElementById("singleResult");
  const resultBadge = document.getElementById("resultBadge");
  const resultProb = document.getElementById("resultProb");
  const resultThreshold = document.getElementById("resultThreshold");
  const reasonsList = document.getElementById("reasonsList");

  singleForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    singleError.textContent = "";
    singleResult.hidden = true;

    const amount = document.getElementById("amount").value;
    const sender_country = document.getElementById("sender_country").value.trim();
    const receiver_country = document.getElementById("receiver_country").value.trim();
    if (!amount || !sender_country || !receiver_country) {
      singleError.textContent = "Amount, sender country, and receiver country are required.";
      return;
    }
    const numAmount = parseFloat(amount);
    if (isNaN(numAmount) || numAmount <= 0) {
      singleError.textContent = "Amount must be a positive number.";
      return;
    }

    const payload = {
      amount: numAmount,
      sender_country,
      receiver_country,
    };
    const hour = document.getElementById("hour_of_day").value;
    const day = document.getElementById("day_of_week").value;
    const account_age = document.getElementById("account_age_days").value;
    const velocity = document.getElementById("transaction_velocity").value;
    const ip_match = document.getElementById("ip_country_matches_sender").value;
    const typos = document.getElementById("message_has_typos").value;
    const msg_type = document.getElementById("message_type").value.trim();
    if (hour !== "") payload.hour_of_day = parseInt(hour, 10);
    if (day !== "") payload.day_of_week = parseInt(day, 10);
    if (account_age !== "") payload.account_age_days = parseInt(account_age, 10);
    if (velocity !== "") payload.transaction_velocity = parseInt(velocity, 10);
    if (ip_match !== "") payload.ip_country_matches_sender = parseInt(ip_match, 10);
    if (typos !== "") payload.message_has_typos = parseInt(typos, 10);
    if (msg_type !== "") payload.message_type = msg_type;

    const threshold = thresholdEl ? parseFloat(thresholdEl.value) : 0.5;
    try {
      const res = await fetch("/api/predict?threshold=" + encodeURIComponent(threshold), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        singleError.textContent = data.error || "Request failed.";
        return;
      }
      singleResult.hidden = false;
      resultProb.textContent = "Fraud probability: " + (data.fraud_probability * 100).toFixed(1) + "%";
      resultThreshold.textContent =
        "Threshold used: " + (data.threshold_used * 100).toFixed(0) + "% — " + (data.is_fraud ? "Flagged as fraud." : "Not flagged.");
      resultBadge.textContent = data.is_fraud ? "Fraud" : "Normal";
      resultBadge.className = "result-badge " + (data.is_fraud ? "fraud" : "normal");
      reasonsList.innerHTML = "";
      if (data.reasons && data.reasons.length) {
        const h4 = document.createElement("h4");
        h4.textContent = "Why this result?";
        reasonsList.appendChild(h4);
        const ul = document.createElement("ul");
        data.reasons.forEach(function (r) {
          const li = document.createElement("li");
          li.textContent = r;
          ul.appendChild(li);
        });
        reasonsList.appendChild(ul);
      }
    } catch (err) {
      singleError.textContent = "Network error: " + err.message;
    }
  });

  // Batch form
  const batchForm = document.getElementById("batchForm");
  const batchError = document.getElementById("batchError");
  const batchResult = document.getElementById("batchResult");
  const batchSuccess = document.getElementById("batchSuccess");

  batchForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    batchError.textContent = "";
    batchResult.hidden = true;
    const fileInput = document.getElementById("batchFile");
    if (!fileInput.files.length) {
      batchError.textContent = "Please choose a CSV file.";
      return;
    }
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("threshold", thresholdEl ? thresholdEl.value : "0.5");
    try {
      const res = await fetch("/api/batch", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const data = await res.json().catch(function () { return {}; });
        batchError.textContent = data.error || "Upload failed.";
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "fraud_scores.csv";
      a.click();
      URL.revokeObjectURL(url);
      batchResult.hidden = false;
      batchSuccess.textContent = "Scored CSV downloaded as fraud_scores.csv. Open it to see fraud_probability, predicted_fraud, and reasons per row.";
    } catch (err) {
      batchError.textContent = "Network error: " + err.message;
    }
  });
})();
