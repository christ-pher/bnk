package server_test

import (
	"testing"

	"vpnmesh/internal/acl"
	"vpnmesh/internal/netmap"
)

func TestSetPolicyPushesCompiledRulesToTargets(t *testing.T) {
	e := startServer(t)
	idL, idN := ident32(t, 1), ident32(t, 2)
	e.enroll(t, "laptop", idL.pub)
	b := e.enroll(t, "nas", idN.pub)

	_, nmsLaptop := dialSession(t, e, idL.priv)
	_, nmsNas := dialSession(t, e, idN.priv)

	err := e.srv.SetPolicy(&acl.Policy{
		Groups: map[string][]string{"trusted": {"laptop"}},
		Rules:  []acl.Rule{{From: []string{"group:trusted"}, To: []string{"nas"}, Allow: []string{"tcp/22"}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	nm := waitNetmap(t, nmsNas, func(nm netmap.Netmap) bool {
		return nm.FilterEnabled && len(nm.Filter) == 1
	})
	r := nm.Filter[0]
	if r.Proto != "tcp" || len(r.Ports) != 1 || r.Ports[0].First != 22 {
		t.Errorf("nas rule = %+v, want tcp/22", r)
	}

	// laptop gets enforcement enabled but no inbound allowances.
	nmL := waitNetmap(t, nmsLaptop, func(nm netmap.Netmap) bool { return nm.FilterEnabled })
	if len(nmL.Filter) != 0 {
		t.Errorf("laptop rules = %+v, want none", nmL.Filter)
	}
	_ = b
}

func TestSetPolicyRejectsInvalid(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "laptop", key32(1))
	err := e.srv.SetPolicy(&acl.Policy{
		Rules: []acl.Rule{{From: []string{"group:ghosts"}, To: []string{"laptop"}, Allow: []string{"tcp/22"}}},
	})
	if err == nil {
		t.Error("invalid policy accepted")
	}
}
