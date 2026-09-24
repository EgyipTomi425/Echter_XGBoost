module;

#include <memory>

module echter.xgb;

import echter.xgb.reg;

namespace echter::xgb
{

struct XGBoost::Impl
{
    Regression regression;
};

XGBoost::XGBoost()
    : impl_(std::make_unique<Impl>())
{
}

XGBoost::~XGBoost() = default;
XGBoost::XGBoost(XGBoost&&) noexcept = default;
XGBoost& XGBoost::operator=(XGBoost&&) noexcept = default;

Regression& XGBoost::regression() noexcept
{
    return impl_->regression;
}

const Regression& XGBoost::regression() const noexcept
{
    return impl_->regression;
}

}
