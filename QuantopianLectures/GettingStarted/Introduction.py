# simple intoduction strategy that says if the 10-day moving average is different than the 30-day moving average most likly there will be a change in rrice that will trade back to its mean


def initialize(context):
    context.security_lists = [sid(5060), sid(7792), sid(24556), sid(1746)]

    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())

    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())


def compute_weights(context, data):
    hist = data.history(context.security_lists, 'price', 30, '1d')

    prices_10 = hist[-10:]
    prices_30 = hist[-30:]

    sma_10 = prices_10.mean()
    sma_30 = prices_30.mean()

    raw_weights = (sma_10 - sma_30) / sma_30

    normalized_weights = raw_weights / raw_weights.abs().sum()

    return normalized_weights


def rebalance(context, data):
    # part of algorithm that acutally places orders

    weights = compute_weights(context, data)

    for security in context.security_lists:
        if data.can_trade(security):
            order_target_percent(security, weights[security])


def record_vars(context, data):
    longs = shorts = 0

    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        if position.amount < 0:
            shorts += 1

    record(leverage=context.account.leverage, long_count=longs, short_count=shorts)


