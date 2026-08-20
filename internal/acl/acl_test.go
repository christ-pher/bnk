package acl

import (
	"net/netip"
	"strings"
	"testing"
)

var nodes = []NodeInfo{
	{Name: "laptop", IP: netip.MustParseAddr("100.64.0.1")},
	{Name: "desktop", IP: netip.MustParseAddr("100.64.0.2")},
	{Name: "nas", IP: netip.MustParseAddr("100.64.0.3")},
}

func TestCompileGroupRuleProducesInboundRulesForTargets(t *testing.T) {
	p := Policy{
		Groups: map[string][]string{
			"trusted": {"laptop", "desktop"},
			"servers": {"nas"},
		},
		Rules: []Rule{
			{From: []string{"group:trusted"}, To: []string{"group:servers"}, Allow: []string{"tcp/22", "tcp/443", "icmp"}},
		},
	}
	compiled, err := Compile(p, nodes)
	if err != nil {
		t.Fatal(err)
	}

	nasRules := compiled["nas"]
	if len(nasRules) != 3 {
		t.Fatalf("nas got %d rules, want 3 (tcp/22, tcp/443, icmp): %+v", len(nasRules), nasRules)
	}
	for _, r := range nasRules {
		if len(r.Srcs) != 2 {
			t.Errorf("rule %+v has %d srcs, want laptop+desktop", r, len(r.Srcs))
		}
	}
	if len(compiled["laptop"]) != 0 || len(compiled["desktop"]) != 0 {
		t.Errorf("non-target nodes got inbound rules: %+v", compiled)
	}
}

func TestCompileWildcardAndPortRange(t *testing.T) {
	p := Policy{Rules: []Rule{
		{From: []string{"*"}, To: []string{"nas"}, Allow: []string{"udp/6000-6100"}},
	}}
	compiled, err := Compile(p, nodes)
	if err != nil {
		t.Fatal(err)
	}
	rules := compiled["nas"]
	if len(rules) != 1 {
		t.Fatalf("rules = %+v", rules)
	}
	r := rules[0]
	if r.Proto != "udp" || len(r.Ports) != 1 || r.Ports[0] != (PortRange{6000, 6100}) {
		t.Errorf("rule = %+v, want udp 6000-6100", r)
	}
	// "*" includes every node except the destination itself.
	if len(r.Srcs) != 2 {
		t.Errorf("wildcard srcs = %v, want the two non-nas nodes", r.Srcs)
	}
	for _, s := range r.Srcs {
		if s.Addr() == netip.MustParseAddr("100.64.0.3") {
			t.Error("wildcard source includes the destination itself")
		}
	}
}

func TestCompileRejectsUnknownGroupAndBadPort(t *testing.T) {
	if _, err := Compile(Policy{Rules: []Rule{{From: []string{"group:nope"}, To: []string{"nas"}, Allow: []string{"tcp/22"}}}}, nodes); err == nil || !strings.Contains(err.Error(), "nope") {
		t.Errorf("unknown group: err = %v", err)
	}
	if _, err := Compile(Policy{Rules: []Rule{{From: []string{"*"}, To: []string{"nas"}, Allow: []string{"tcp/notaport"}}}}, nodes); err == nil {
		t.Error("bad port accepted")
	}
	if _, err := Compile(Policy{Rules: []Rule{{From: []string{"ghost"}, To: []string{"nas"}, Allow: []string{"tcp/22"}}}}, nodes); err == nil {
		t.Error("unknown node name accepted")
	}
}

func TestCompiledRuleMatch(t *testing.T) {
	r := CompiledRule{
		Srcs:  []netip.Prefix{netip.MustParsePrefix("100.64.0.1/32")},
		Proto: "tcp",
		Ports: []PortRange{{22, 22}, {8000, 9000}},
	}
	cases := []struct {
		src   string
		proto string
		port  uint16
		want  bool
	}{
		{"100.64.0.1", "tcp", 22, true},
		{"100.64.0.1", "tcp", 8500, true},
		{"100.64.0.1", "tcp", 23, false},
		{"100.64.0.1", "udp", 22, false},
		{"100.64.0.9", "tcp", 22, false},
	}
	for _, c := range cases {
		got := r.Match(netip.MustParseAddr(c.src), c.proto, c.port)
		if got != c.want {
			t.Errorf("Match(%s %s/%d) = %v, want %v", c.src, c.proto, c.port, got, c.want)
		}
	}
}

func TestICMPRuleIgnoresPorts(t *testing.T) {
	r := CompiledRule{Srcs: []netip.Prefix{netip.MustParsePrefix("100.64.0.0/10")}, Proto: "icmp"}
	if !r.Match(netip.MustParseAddr("100.64.0.1"), "icmp", 0) {
		t.Error("icmp rule did not match")
	}
}
