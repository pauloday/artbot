defmodule ArtshopWeb.PageController do
  use ArtshopWeb, :controller

  def index(conn, _params) do
    render(conn, "index.html")
  end
end
