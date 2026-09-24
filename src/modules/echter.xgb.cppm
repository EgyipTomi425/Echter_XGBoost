export module echter.xgb;

export import echter.xgb.reg;
export import echter.xgb.io;

export namespace echter::xgb
{

class XGBoost
{
public:
    [[nodiscard]] Regression& regression() noexcept
    {
        return regression_;
    }

    [[nodiscard]] const Regression& regression() const noexcept
    {
        return regression_;
    }

private:
    Regression regression_;
};

}
