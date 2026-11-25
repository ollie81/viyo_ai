-- NardoPay: match payments to orders by the payment link's code. Run after nardopay_orders.sql.
-- Safe to run more than once. Adds one column and one index; no data changes.
--
-- NardoPay doesn't send a link's metadata back in its payment webhook, but it does send link_code: the code in the
-- link it returned (https://nardopay.com/p/<link_code>?pay=1). VIYO saves that code on the order when it creates the
-- link, and finds the order by it when the payment arrives. One link code belongs to one order, ever.
alter table public.nardopay_orders add column if not exists link_code text;
create unique index if not exists nardopay_orders_link_code_key
  on public.nardopay_orders (link_code) where link_code is not null;

do $$ begin
  alter table public.nardopay_orders add constraint nardopay_orders_link_code_check
    check (link_code is null or link_code ~ '^[A-Za-z0-9_-]{6,64}$');
exception when duplicate_object then null; end $$;
