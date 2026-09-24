module;

#include <memory>

export module echter.xgb;

export import echter;
export import echter.xgb.reg;
export import echter.xgb.io;

export namespace echter::xgb
{

class XGBoost
{
public:
    XGBoost();
    ~XGBoost();

    XGBoost(XGBoost&&) noexcept;
    XGBoost& operator=(XGBoost&&) noexcept;

    XGBoost(const XGBoost&) = delete;
    XGBoost& operator=(const XGBoost&) = delete;

    [[nodiscard]] Regression& regression() noexcept;
    [[nodiscard]] const Regression& regression() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
