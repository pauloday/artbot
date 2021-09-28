defmodule Artshop.Repo do
  use Ecto.Repo,
    otp_app: :artshop,
    adapter: Ecto.Adapters.Postgres
end
