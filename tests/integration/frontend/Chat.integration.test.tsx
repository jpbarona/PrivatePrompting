import {
  afterEach,
  describe,
  expect,
  it,
  React,
  render,
  screen,
  userEvent,
  vi,
  waitFor,
} from "./testing";

import { Chat } from "../../../frontend/src/app/components/Chat";

describe("Chat <-> /infer integration contract", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends prompt to /infer and renders host response", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ response: "Host reply" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    );

    render(React.createElement(Chat));

    await user.type(screen.getByPlaceholderText("Type a message..."), "hello from UI");
    await user.click(screen.getByRole("button"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "hello from UI" }),
    });
    expect(await screen.findByText("Host reply")).toBeInTheDocument();
  });

  it("renders backend validation message returned by /infer", async () => {
    const user = userEvent.setup();
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "prompt must not be empty" }), {
        status: 422,
        headers: { "Content-Type": "application/json" },
      })
    );

    render(React.createElement(Chat));

    await user.type(screen.getByPlaceholderText("Type a message..."), "invalid payload");
    await user.click(screen.getByRole("button"));

    expect(await screen.findByText("Error: prompt must not be empty")).toBeInTheDocument();
  });
});
