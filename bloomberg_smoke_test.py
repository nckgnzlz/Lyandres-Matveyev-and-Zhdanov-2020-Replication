"""Pull one sample of every Bloomberg input the LMZ pipeline needs, for AAPL US Equity."""
import blpapi

TICKER = "AAPL US Equity"
s = blpapi.Session(blpapi.SessionOptions())
assert s.start() and s.openService("//blp/refdata"), "cannot connect"
svc = s.getService("//blp/refdata")


def run(req):
    s.sendRequest(req)
    out = []
    while True:
        ev = s.nextEvent(30000)
        for msg in ev:
            if msg.messageType() in (blpapi.Name("ReferenceDataResponse"), blpapi.Name("HistoricalDataResponse")):
                out.append(msg)
        if ev.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.TIMEOUT):
            return out


def ref(fields, overrides=None, security=TICKER):
    req = svc.createRequest("ReferenceDataRequest")
    req.append("securities", security)
    for f in fields:
        req.append("fields", f)
    for k, v in (overrides or {}).items():
        o = req.getElement("overrides").appendElement()
        o.setElement("fieldId", k)
        o.setElement("value", v)
    return run(req)


def hist(fields, start, end, per):
    req = svc.createRequest("HistoricalDataRequest")
    req.append("securities", TICKER)
    for f in fields:
        req.append("fields", f)
    req.set("startDate", start)
    req.set("endDate", end)
    req.set("periodicitySelection", per)
    return run(req)


def show_hist(title, msgs, maxrows=6):
    print(f"\n=== {title}")
    for m in msgs:
        sd = m.getElement("securityData")
        if sd.hasElement("securityError"):
            print("SECURITY ERROR:", sd.getElement("securityError"))
        if sd.hasElement("fieldExceptions") and sd.getElement("fieldExceptions").numValues():
            print("FIELD EXCEPTIONS:", sd.getElement("fieldExceptions"))
        fd = sd.getElement("fieldData")
        n = fd.numValues()
        print(f"rows returned: {n}")
        for i in list(range(min(maxrows // 2, n))) + list(range(max(n - maxrows // 2, maxrows // 2), n)):
            row = fd.getValueAsElement(i)
            print("  " + ", ".join(f"{row.getElement(j).name()}={row.getElement(j).getValue()}" for j in range(row.numElements())))


# 1. Universe: historical Russell 3000 membership (bulk field with date override)
print("=== Universe: RAY Index members as of 2010-12-31 (INDX_MWEIGHT_HIST)")
for m in ref(["INDX_MWEIGHT_HIST"], {"END_DATE_OVERRIDE": "20101231"}, security="RAY Index"):
    sd = m.getElement("securityData").getValueAsElement(0)
    if sd.getElement("fieldExceptions").numValues():
        print("FIELD EXCEPTIONS:", sd.getElement("fieldExceptions"))
    fd = sd.getElement("fieldData")
    if fd.hasElement("INDX_MWEIGHT_HIST"):
        bulk = fd.getElement("INDX_MWEIGHT_HIST")
        names = [bulk.getValueAsElement(i).getElement(0).getValueAsString() for i in range(bulk.numValues())]
        print(f"members returned: {len(names)}; first 5: {names[:5]}; AAPL included: {any(n.startswith('AAPL') for n in names)}")

# 2. Static
print("\n=== Static (current values)")
for m in ref(["GICS_INDUSTRY", "GICS_SECTOR", "CRNCY", "EQY_FUND_CRNCY"]):
    print(m.getElement("securityData").getValueAsElement(0).getElement("fieldData"))

# 3. Monthly market data + LTG history
show_hist("Monthly 2010-2014: CUR_MKT_CAP, TOT_RETURN_INDEX_GROSS_DVDS",
          hist(["CUR_MKT_CAP", "TOT_RETURN_INDEX_GROSS_DVDS"], "20100101", "20141231", "MONTHLY"))
show_hist("Monthly 2010-2014: BEST_EST_LONG_TERM_GROWTH (history)",
          hist(["BEST_EST_LONG_TERM_GROWTH"], "20100101", "20141231", "MONTHLY"))

# 4. Annual fundamentals
show_hist("Annual 2006-2014: PP&E gross, EBITDA, D&A, ST debt, LT debt",
          hist(["BS_GROSS_FIX_ASSET", "EBITDA", "CF_DEPR_AMORT", "BS_ST_BORROW", "BS_LT_BORROW"],
               "20060101", "20141231", "YEARLY"), maxrows=10)

# 5. Quarterly sales
show_hist("Quarterly 2006-2014: SALES_REV_TURN",
          hist(["SALES_REV_TURN"], "20060101", "20141231", "QUARTERLY"))

s.stop()
