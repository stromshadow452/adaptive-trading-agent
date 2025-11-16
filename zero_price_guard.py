# zero_price_guard.py
def guard(order, logger, artifact_orders):
    """
    Return True to proceed with execution, False to skip (and record artifact).
    logger may be None; artifact_orders is expected to be a list.
    """
    try:
        price_raw = order.get("price", 0)
        try:
            price = float(price_raw or 0)
        except Exception:
            # try parse strings like "0.00000000"
            try:
                price = float(str(price_raw).strip())
            except Exception:
                price = 0.0
        if price == 0.0:
            msg = "SKIP_EXEC zero-price order -> %s" % (order,)
            if logger is not None and hasattr(logger, "warn"):
                try:
                    logger.warn(msg)
                except Exception:
                    print(msg)
            else:
                print(msg)
            order["status"] = "SKIPPED_ZERO_PRICE"
            try:
                if isinstance(artifact_orders, list):
                    artifact_orders.append(order)
            except Exception:
                pass
            return False
    except Exception as e:
        err = "zero-price-guard error -> %s" % (e,)
        if logger is not None and hasattr(logger, "warn"):
            try:
                logger.warn(err)
            except Exception:
                print(err)
        else:
            print(err)
    return True
